/*
 * CUDA Expectation Maximization with Gaussian Mixture Models
 * Multi-GPU implemenetation using OpenMP
 *
 */

// Includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h> // for clock(), clock_t, CLOCKS_PER_SEC
#include <omp.h>
#include <chrono> 

// Includes, project
#include "gaussian.h"
#include "invert_matrix.h"

// Includes, kernels
#include "gaussian_kernel.cu"

// Function prototypes
extern "C" float* readData(char* f, int* ndims, int*nevents);
int validateArguments(int argc, char** argv, int* num_clusters, int* target_num_clusters);
void writeCluster(FILE* f, clusters_t clusters, int c,  int num_dimensions);
void printCluster(clusters_t clusters, int c, int num_dimensions);
float cluster_distance(clusters_t clusters, int c1, int c2, clusters_t temp_cluster, int num_dimensions);
void copy_cluster(clusters_t dest, int c_dest, clusters_t src, int c_src, int num_dimensions);
void add_clusters(clusters_t clusters, int c1, int c2, clusters_t temp_cluster, int num_dimensions);

/*
 * Seeds the cluster centers (means) with random data points
 */
void seed_clusters(clusters_t* clusters, float* fcs_data, int num_clusters, int num_dimensions, int num_events) {
    float fraction;
    int seed;
    if(num_clusters > 1) {
        fraction = (num_events-1.0f)/(num_clusters-1.0f);
    } else {
        fraction = 0.0;
    }
    srand((unsigned int) 0);
    // Sets the means from evenly distributed points in the input data
    for(int c=0; c < num_clusters; c++) {
        clusters->N[c] = (float)num_events/(float)num_clusters;
        #if UNIFORM_SEED
            for(int d=0; d < num_dimensions; d++)
                clusters->means[c*num_dimensions+d] = fcs_data[((int)(c*fraction))*num_dimensions+d];
        #else
            seed = rand() % num_events; 
            DEBUG("Cluster %d seed = event #%d\n",c,seed);
            for(int d=0; d < num_dimensions; d++)
                clusters->means[c*num_dimensions+d] = fcs_data[seed*num_dimensions+d];
        #endif
    }
}

/*
 * Main function that contains all the logic of the benchmark
 */
clusters_t* cluster(int original_num_clusters, int desired_num_clusters, int* final_num_clusters, int num_dimensions, int num_events, float* fcs_data_by_event) {
	int ideal_num_clusters;
	int stop_number;

	// Number of clusters to stop iterating at.
    if(desired_num_clusters == 0) {
        stop_number = 1;
    } else {
        stop_number = desired_num_clusters;
    }

    // Number of GPUs
    int num_gpus;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&num_gpus));

    if (num_gpus < 1) {
        printf("ERROR: No CUDA capable GPUs detected.\n");
        return NULL;
    } else if(num_gpus == 1) {
        printf("Warning: Only 1 CUDA GPU detected. Running single GPU version would be more efficient.\n");
    } else {
        PRINT("Using %d Host-threads with %d GPUs\n",num_gpus,num_gpus);
    }
	
	// Transpose the event data (allows coalesced access pattern in E-step kernel)
    // This has consecutive values being from the same dimension of the data 
    // (num_dimensions by num_events matrix)
    float* fcs_data_by_dimension  = (float*) malloc(sizeof(float)*num_events*num_dimensions);
    
    for(int e=0; e<num_events; e++) {
        for(int d=0; d<num_dimensions; d++) {
            if(isnan(fcs_data_by_event[e*num_dimensions+d])) {
                printf("Error: Found NaN value in input data. Exiting.\n");
                return NULL;
            }
            fcs_data_by_dimension[d*num_events+e] = fcs_data_by_event[e*num_dimensions+d];
        }
    }    
   
    PRINT("Number of events: %d\n",num_events);
    PRINT("Number of dimensions: %d\n\n",num_dimensions);
    PRINT("Starting with %d cluster(s), will stop at %d cluster(s).\n",original_num_clusters,stop_number);
   
    // Setup the cluster data structures shared between the GPUs and CPU (Unified Memory)
    clusters_t* um_clusters;
    CUDA_SAFE_CALL(cudaMallocManaged(&um_clusters, sizeof(clusters_t)*num_gpus));
    for (int g = 0; g< num_gpus; g++){
        CUDA_SAFE_CALL(cudaMallocManaged(&(um_clusters[g].N), sizeof(float)*original_num_clusters));
        CUDA_SAFE_CALL(cudaMallocManaged(&(um_clusters[g].pi), sizeof(float)*original_num_clusters));
        CUDA_SAFE_CALL(cudaMallocManaged(&(um_clusters[g].constant), sizeof(float)*original_num_clusters));
        CUDA_SAFE_CALL(cudaMallocManaged(&(um_clusters[g].avgvar), sizeof(float)*original_num_clusters));
        CUDA_SAFE_CALL(cudaMallocManaged(&(um_clusters[g].means), sizeof(float)*num_dimensions*original_num_clusters));
        CUDA_SAFE_CALL(cudaMallocManaged(&(um_clusters[g].R), sizeof(float)*num_dimensions*num_dimensions*original_num_clusters));
        CUDA_SAFE_CALL(cudaMallocManaged(&(um_clusters[g].Rinv), sizeof(float)*num_dimensions*num_dimensions*original_num_clusters));
        CUDA_SAFE_CALL(cudaMallocManaged(&(um_clusters[g].memberships), sizeof(float)*original_num_clusters*num_events*2));  
    }
    
    // Declare another set of clusters for saving the results of the best configuration
    clusters_t* saved_clusters = (clusters_t*) malloc(sizeof(clusters_t));
    saved_clusters->N = (float*) malloc(sizeof(float)*original_num_clusters);
    saved_clusters->pi = (float*) malloc(sizeof(float)*original_num_clusters);
    saved_clusters->constant = (float*) malloc(sizeof(float)*original_num_clusters);
    saved_clusters->avgvar = (float*) malloc(sizeof(float)*original_num_clusters);
    saved_clusters->means = (float*) malloc(sizeof(float)*num_dimensions*original_num_clusters);
    saved_clusters->R = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*original_num_clusters);
    saved_clusters->Rinv = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*original_num_clusters);
    saved_clusters->memberships = (float*) malloc(sizeof(float)*num_events*original_num_clusters);
    if(!saved_clusters->means || !saved_clusters->R || !saved_clusters->Rinv || !saved_clusters->memberships) { 
        printf("ERROR: Could not allocate memory for clusters.\n"); 
        return NULL; 
    }
    DEBUG("Finished allocating shared cluster structures on host\n");
        
    // Used to hold the result from regroup kernel (Unified Memory)
    float* shared_likelihoods;
    CUDA_SAFE_CALL(cudaMallocManaged(&shared_likelihoods, sizeof(float)*NUM_BLOCKS*num_gpus));
    float likelihood, old_likelihood;
    float min_rissanen;
    
    // Main thread splits into one thread per GPU at this point
    omp_set_num_threads(num_gpus);
    #pragma omp parallel shared(um_clusters,fcs_data_by_event,fcs_data_by_dimension,shared_likelihoods,likelihood,old_likelihood,ideal_num_clusters,min_rissanen) 
    {
        // Set the device for this thread
        unsigned int tid  = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();

        // Get information about the GPU for rhis thread
        cudaSetDevice(tid);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, tid);
        printf("CPU thread %d (of %d) using device %d: %s\n", tid, num_gpus, tid, prop.name);
        
        // Used as a temporary cluster for combining clusters in "distance" computations
        clusters_t scratch_cluster;
        scratch_cluster.N = (float*) malloc(sizeof(float));
        scratch_cluster.pi = (float*) malloc(sizeof(float));
        scratch_cluster.constant = (float*) malloc(sizeof(float));
        scratch_cluster.avgvar = (float*) malloc(sizeof(float));
        scratch_cluster.means = (float*) malloc(sizeof(float)*num_dimensions);
        scratch_cluster.R = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions);
        scratch_cluster.Rinv = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions);
        scratch_cluster.memberships = (float*) malloc(sizeof(float)*num_events);

        DEBUG("Finished allocating memory on host for clusters.\n");
        
        // Determine how many events this GPU will handle
        int events_per_gpu = num_events / num_gpus;
        int my_num_events = events_per_gpu;
        if(tid == num_gpus-1) {
            my_num_events += num_events % num_gpus; // last gpu has to handle the remaining uneven events
        }

        DEBUG("GPU %d will handle %d events\n",tid,my_num_events);     

        // Allocate memory for FCS data for this thread (Unified Memory)
        float* um_fcs_data_by_event;
        float* um_fcs_data_by_dimension;

        int mem_size = num_dimensions * my_num_events * sizeof(float);
        CUDA_SAFE_CALL(cudaMallocManaged(&um_fcs_data_by_event, mem_size));
        CUDA_SAFE_CALL(cudaMallocManaged(&um_fcs_data_by_dimension, mem_size));

        // Copying data from general data structure to each thread
        memcpy(um_fcs_data_by_event, &fcs_data_by_event[num_dimensions*events_per_gpu*tid], mem_size);

        // Copying the transposed data is trickier since it's not all contigious for the relavant events
        for(int d=0; d < num_dimensions; d++) {
            memcpy(&um_fcs_data_by_dimension[d*my_num_events], &fcs_data_by_dimension[d*num_events + tid*events_per_gpu], sizeof(float)*my_num_events);
        }
        cudaDeviceSynchronize();
 
        DEBUG("GPU %d: Finished copying FCS data to device.\n",tid);
        
        //////////////// Initialization done, starting kernels //////////////// 
        DEBUG("Invoking seed_clusters kernel.\n");

        // seed_clusters sets initial pi values, 
        // finds the means / covariances and copies it to all the clusters
        #pragma omp master
        {
            //  TODO: seed_clusters can't be done on gpu since it doesnt have all the events
            //  Just have host pick random events for the means and use identity matrix for covariance
            //  Only tricky part is how to do average variance? 
            //  Make a kernel for that and reduce on host like the means/covariance?
            seed_clusters<<< 1, NUM_THREADS_MSTEP >>>(um_fcs_data_by_event, um_clusters, num_dimensions, original_num_clusters, my_num_events);
            cudaDeviceSynchronize();
            
            DEBUG("Invoking constants kernel.\n");
            // Computes the R matrix inverses, and the gaussian constant
            constants_kernel<<<original_num_clusters, NUM_THREADS_MSTEP>>>(um_clusters, original_num_clusters, num_dimensions);
            cudaDeviceSynchronize();

            seed_clusters(&um_clusters[0], fcs_data_by_event, original_num_clusters, num_dimensions, num_events);
            DEBUG("Starting Clusters\n");
            if (ENABLE_DEBUG) {
                for(int c=0; c < original_num_clusters; c++) {
                    DEBUG("Cluster #%d\n",c);
    
                    DEBUG("\tN: %f\n",um_clusters[0].N[c]); 
                    DEBUG("\tpi: %f\n",um_clusters[0].pi[c]); 
    
                    // means
                    DEBUG("\tMeans: ");
                    for(int d=0; d < num_dimensions; d++) {
                        DEBUG("%.2f ",um_clusters[0].means[c*num_dimensions+d]);
                    }
                    DEBUG("\n");
    
                    DEBUG("\tR:\n\t");
                    for(int d=0; d < num_dimensions; d++) {
                        for(int e=0; e < num_dimensions; e++)
                            DEBUG("%.2f ",um_clusters[0].R[c*num_dimensions*num_dimensions+d*num_dimensions+e]);
                        DEBUG("\n\t");
                    }
                    DEBUG("R-inverse:\n\t");
                    for(int d=0; d < num_dimensions; d++) {
                        for(int e=0; e < num_dimensions; e++)
                            DEBUG("%.2f ",um_clusters[0].Rinv[c*num_dimensions*num_dimensions+d*num_dimensions+e]);
                        DEBUG("\n\t");
                    }
                    DEBUG("\n");
                    DEBUG("\tAvgvar: %e\n",um_clusters[0].avgvar[c]);
                    DEBUG("\tConstant: %e\n",um_clusters[0].constant[c]);
                }
            }
            
        }

        // Synchronize after first gpu does the seeding, copy result to all gpus
        #pragma omp barrier
        memcpy(um_clusters[tid].N, um_clusters[0].N, sizeof(float)*original_num_clusters);
        memcpy(um_clusters[tid].pi, um_clusters[0].pi, sizeof(float)*original_num_clusters);
        memcpy(um_clusters[tid].constant, um_clusters[0].constant, sizeof(float)*original_num_clusters);
        memcpy(um_clusters[tid].avgvar, um_clusters[0].avgvar, sizeof(float)*original_num_clusters);
        memcpy(um_clusters[tid].means, um_clusters[0].means, sizeof(float)*num_dimensions*original_num_clusters);
        memcpy(um_clusters[tid].R, um_clusters[0].R, sizeof(float)*num_dimensions*num_dimensions*original_num_clusters);
        memcpy(um_clusters[tid].Rinv, um_clusters[0].Rinv, sizeof(float)*num_dimensions*num_dimensions*original_num_clusters);
       
        // Calculate an epsilon value
        float epsilon = (1+num_dimensions+0.5f*(num_dimensions+1)*num_dimensions)*logf((float)num_events*num_dimensions)*0.001f;
        int iters;
        
        #pragma omp master
        {
            PRINT("Gaussian.cu: epsilon = %f\n",epsilon);
        }
        
        // Variables for GMM reduce order
        float distance, min_distance = 0.0;
        float rissanen;
        int min_c1, min_c2;

        // Timer for exec time
        #pragma omp barrier
        auto start1 = std::chrono::steady_clock::now();
         
        for(int num_clusters=original_num_clusters; num_clusters >= stop_number; num_clusters--) {
            /*************** EM ALGORITHM *****************************/
            
            // Do initial E-step
            // Calculates a cluster membership probability
            // for each event and each cluster.
            DEBUG("Invoking E-step kernels.");
            estep1<<<dim3(num_clusters,NUM_BLOCKS), NUM_THREADS_ESTEP>>>(um_fcs_data_by_dimension, &(um_clusters[tid]), num_dimensions, my_num_events);
            estep2<<<NUM_BLOCKS, NUM_THREADS_ESTEP>>>(&(um_clusters[tid]), num_dimensions, num_clusters, my_num_events, &shared_likelihoods[tid*NUM_BLOCKS]);
            cudaDeviceSynchronize();

            #pragma omp barrier

            #pragma omp master
            {
                likelihood = 0.0;
                for(int i=0;i<NUM_BLOCKS*num_gpus;i++) {
                    likelihood += shared_likelihoods[i]; 
                }
                DEBUG("Likelihood: %e\n",likelihood);
            }
            #pragma omp barrier

            float change = epsilon*2;
            
            #pragma omp master
            {
                PRINT("Performing EM algorithm on %d clusters.\n", num_clusters);
            }
            iters = 0;
            // This is the iterative loop for the EM algorithm.
            // It re-estimates parameters, re-computes constants, and then regroups the events
            // These steps keep repeating until the change in likelihood is less than some epsilon        
            while(iters < MIN_ITERS || (fabs(change) > epsilon && iters < MAX_ITERS)) {
                #pragma omp master
                {
                    old_likelihood = likelihood;
                }
                
                DEBUG("Invoking reestimate_parameters (M-step) kernel.");
                // This kernel computes a new N, pi isn't updated until compute_constants though
                mstep_N<<<num_clusters, NUM_THREADS_MSTEP>>>(&(um_clusters[tid]), num_dimensions, num_clusters, my_num_events);
                cudaDeviceSynchronize();
                
                // Reduce N for all clusters, copy back to device
                #pragma omp barrier

                #pragma omp master
                {
                    for(int g=1; g < num_gpus; g++) {
                        for(int c=0; c < num_clusters; c++) {
                            um_clusters[0].N[c] += um_clusters[g].N[c];
                            DEBUG("Cluster %d: N = %f\n",c,um_clusters[0].N[c]);
                        }
                    }
                }
                #pragma omp barrier
                memcpy(um_clusters[tid].N, um_clusters[0].N, sizeof(float)*num_clusters);

                dim3 gridDim1(num_clusters,num_dimensions);
                mstep_means<<<gridDim1, NUM_THREADS_MSTEP>>>(um_fcs_data_by_dimension, &(um_clusters[tid]), num_dimensions, num_clusters, my_num_events);
                cudaDeviceSynchronize();
                
                // Reduce means for all clusters, copy back to device
                #pragma omp barrier

                #pragma omp master
                {
                    for(int g=1; g < num_gpus; g++) {
                        for(int c=0; c < num_clusters; c++) {
                            for(int d=0; d < num_dimensions; d++) {
                                um_clusters[0].means[c*num_dimensions+d] += um_clusters[g].means[c*num_dimensions+d];
                            }
                        }
                    }
                    for(int c=0; c < num_clusters; c++) {
                        DEBUG("Cluster %d  Means:",c,um_clusters[0].N[c]);
                        for(int d=0; d < num_dimensions; d++) {
                            if(um_clusters[0].N[c] > 0.5f) {
                                um_clusters[0].means[c*num_dimensions+d] /= um_clusters[0].N[c];
                            } else {
                                um_clusters[0].means[c*num_dimensions+d] = 0.0f;
                            }
                            DEBUG(" %f",um_clusters[0].means[c*num_dimensions+d]);
                        }
                        DEBUG("\n");
                    }
                }
                #pragma omp barrier

                memcpy(um_clusters[tid].means, um_clusters[0].means, sizeof(float)*num_clusters*num_dimensions);

                // Covariance is symmetric, so we only need to compute N*(N+1)/2 matrix elements per cluster
                dim3 gridDim2(num_clusters, num_dimensions*(num_dimensions+1)/2);
                mstep_covariance2<<<dim3((num_clusters+NUM_CLUSTERS_PER_BLOCK-1)/NUM_CLUSTERS_PER_BLOCK, num_dimensions*(num_dimensions+1)/2), NUM_THREADS_MSTEP>>>(um_fcs_data_by_dimension, &(um_clusters[tid]), num_dimensions, num_clusters, my_num_events);
                cudaDeviceSynchronize();
                
                // Reduce R for all clusters, copy back to device
                #pragma omp barrier

                #pragma omp master
                {
                    for(int g=1; g < num_gpus; g++) {
                        for(int c=0; c < num_clusters; c++) {
                            for(int d=0; d < num_dimensions*num_dimensions; d++) {
                                um_clusters[0].R[c*num_dimensions*num_dimensions+d] += um_clusters[g].R[c*num_dimensions*num_dimensions+d];
                            }
                        }
                    }
                    for(int c=0; c < num_clusters; c++) {
                        if(um_clusters[0].N[c] > 0.5f) {
                            for(int d=0; d < num_dimensions*num_dimensions; d++) {
                                um_clusters[0].R[c*num_dimensions*num_dimensions+d] /= um_clusters[0].N[c];
                            }
                        } else {
                            for(int i=0; i < num_dimensions; i++) {
                                for(int j=0; j < num_dimensions; j++) {
                                    if(i == j) {
                                        um_clusters[0].R[c*num_dimensions*num_dimensions+i*num_dimensions+j] = 1.0;
                                    } else {
                                        um_clusters[0].R[c*num_dimensions*num_dimensions+i*num_dimensions+j] = 0.0;
                                    }
                                }
                            }
                        }
                    }
                }
                #pragma omp barrier

                memcpy(um_clusters[tid].R, um_clusters[0].R,sizeof(float)*num_clusters*num_dimensions*num_dimensions);
                
                cudaDeviceSynchronize();
                
                DEBUG("Invoking constants kernel.");
                // Inverts the R matrices, computes the constant, normalizes cluster probabilities
                constants_kernel<<<num_clusters, NUM_THREADS_MSTEP>>>(&(um_clusters[tid]),num_clusters,num_dimensions);
                cudaDeviceSynchronize();
                
                #pragma omp master 
                {
                    for(int temp_c=0; temp_c < num_clusters; temp_c++)
                        DEBUG("Cluster %d constant: %e\n", temp_c, um_clusters[tid].constant[temp_c]);
                }

                DEBUG("Invoking regroup (E-step) kernel with %d blocks.\n",NUM_BLOCKS);
                // Compute new cluster membership probabilities for all the events
                estep1<<<dim3(num_clusters,NUM_BLOCKS), NUM_THREADS_ESTEP>>>(um_fcs_data_by_dimension, &(um_clusters[tid]), num_dimensions, my_num_events);
                estep2<<<NUM_BLOCKS, NUM_THREADS_ESTEP>>>(&(um_clusters[tid]), num_dimensions, num_clusters, my_num_events, &shared_likelihoods[tid*NUM_BLOCKS]);
                cudaDeviceSynchronize();

                #pragma omp barrier

                #pragma omp master
                {
                    likelihood = 0.0;
                    for(int i=0;i<NUM_BLOCKS*num_gpus;i++) {
                        likelihood += shared_likelihoods[i]; 
                    }
                    DEBUG("Likelihood: %e\n",likelihood);
                }
                #pragma omp barrier // synchronize for likelihood
                
                change = likelihood - old_likelihood;
                DEBUG("GPU %d, Change in likelihood: %e\n",tid,change);

                iters++;
                #pragma omp barrier // synchronize loop iteration
            }

            DEBUG("GPU %d done with EM loop\n",tid);
            
            for(int c=0; c < num_clusters; c++) {
                memcpy(&(um_clusters[0].memberships[c*num_events+tid*events_per_gpu]), &(um_clusters[tid].memberships[c*my_num_events]),sizeof(float)*my_num_events);
            }
            #pragma omp barrier
            DEBUG("GPU %d done with copying cluster data from device\n",tid); 
            
            // Calculate Rissanen Score
            rissanen = -likelihood + 0.5f*(num_clusters*(1.0f+num_dimensions+0.5f*(num_dimensions+1.0f)*num_dimensions)-1.0f)*logf((float)num_events*num_dimensions);
            #pragma omp master 
            {
                PRINT("\nLikelihood: %e\n",likelihood);
                PRINT("\nRissanen Score: %e\n",rissanen);
            }
            
            #pragma omp barrier
            #pragma omp master
            {
                // Save the cluster data the first time through, so we have a base rissanen score and result
                // Save the cluster data if the solution is better and the user didn't specify a desired number
                // If the num_clusters equals the desired number, stop
                if(num_clusters == original_num_clusters || (rissanen < min_rissanen && desired_num_clusters == 0) || (num_clusters == desired_num_clusters)) {
                    min_rissanen = rissanen;
                    ideal_num_clusters = num_clusters;
                    // Save the cluster configuration somewhere
                    memcpy(saved_clusters->N,um_clusters[0].N,sizeof(float)*num_clusters);
                    memcpy(saved_clusters->pi,um_clusters[0].pi,sizeof(float)*num_clusters);
                    memcpy(saved_clusters->constant,um_clusters[0].constant,sizeof(float)*num_clusters);
                    memcpy(saved_clusters->avgvar,um_clusters[0].avgvar,sizeof(float)*num_clusters);
                    memcpy(saved_clusters->means,um_clusters[0].means,sizeof(float)*num_dimensions*num_clusters);
                    memcpy(saved_clusters->R,um_clusters[0].R,sizeof(float)*num_dimensions*num_dimensions*num_clusters);
                    memcpy(saved_clusters->Rinv,um_clusters[0].Rinv,sizeof(float)*num_dimensions*num_dimensions*num_clusters);
                    memcpy(saved_clusters->memberships,um_clusters[0].memberships,sizeof(float)*num_events*num_clusters);
                }
            }
            #pragma omp barrier

            /**************** Reduce GMM Order ********************/
            // Don't want to reduce order on the last iteration
            if(num_clusters > stop_number) {
                #pragma omp master
                {
                    // First eliminate any "empty" clusters 
                    for(int i=num_clusters-1; i >= 0; i--) {
                        if(um_clusters[0].N[i] < 0.5) {
                            DEBUG("Cluster #%d has less than 1 data point in it.\n",i);
                            for(int j=i; j < num_clusters-1; j++) {
                                copy_cluster(um_clusters[0],j,um_clusters[0],j+1,num_dimensions);
                            }
                            num_clusters--;
                        }
                    }
                    
                    min_c1 = 0;
                    min_c2 = 1;
                    DEBUG("Number of non-empty clusters: %d\n",num_clusters); 
                    // For all combinations of subclasses...
                    // If the number of clusters got really big might need to do a non-exhaustive search
                    // Even with 100*99/2 combinations this doesn't seem to take too long
                    for(int c1=0; c1<num_clusters;c1++) {
                        for(int c2=c1+1; c2<num_clusters;c2++) {
                            // compute distance function between the 2 clusters
                            distance = cluster_distance(um_clusters[0],c1,c2,scratch_cluster,num_dimensions);
                            
                            // Keep track of minimum distance
                            if((c1 ==0 && c2 == 1) || distance < min_distance) {
                                min_distance = distance;
                                min_c1 = c1;
                                min_c2 = c2;
                            }
                        }
                    }

                    PRINT("\nMinimum distance between (%d,%d). Combining clusters\n",min_c1,min_c2);
                    // Add the two clusters with min distance together
                    add_clusters(um_clusters[0],min_c1,min_c2,scratch_cluster,num_dimensions);
                    // Copy new combined cluster into the main group of clusters, compact them
                    copy_cluster(um_clusters[0],min_c1,scratch_cluster,0,num_dimensions);
                    for(int i=min_c2; i < num_clusters-1; i++) {
                        DEBUG("Copying cluster %d to cluster %d\n", i + 1, i);
                        copy_cluster(um_clusters[0],i,um_clusters[0],i+1,num_dimensions);
                    }
                }
                #pragma omp barrier

                // Copy the clusters back to the device
                memcpy(um_clusters[tid].N, um_clusters[0].N, sizeof(float)*num_clusters);
                memcpy(um_clusters[tid].pi, um_clusters[0].pi, sizeof(float)*num_clusters);
                memcpy(um_clusters[tid].constant, um_clusters[0].constant, sizeof(float)*num_clusters);
                memcpy(um_clusters[tid].avgvar, um_clusters[0].avgvar, sizeof(float)*num_clusters);
                memcpy(um_clusters[tid].means, um_clusters[0].means, sizeof(float)*num_dimensions*num_clusters);
                memcpy(um_clusters[tid].R, um_clusters[0].R, sizeof(float)*num_dimensions*num_dimensions*num_clusters);
                memcpy(um_clusters[tid].Rinv, um_clusters[0].Rinv, sizeof(float)*num_dimensions*num_dimensions*num_clusters);

                for(int c=0; c < num_clusters; c++) {
                    memcpy(&um_clusters[tid].memberships[c*my_num_events], &(um_clusters[0].memberships[c*num_events+tid*(num_events/num_gpus)]),sizeof(float)*my_num_events);
                }

            } // GMM reduction block 

            #pragma omp barrier
        } // outer loop from M to 1 clusters
        #pragma omp master
        PRINT("\nFinal rissanen Score was: %f, with %d clusters.\n",min_rissanen,ideal_num_clusters);
        #pragma omp barrier 

        auto diff1 = std::chrono::steady_clock::now() - start1;
        PROFILING("##### Time execution on device %d: %ld micro sec. #####\n", tid, std::chrono::duration_cast<std::chrono::microseconds>(diff1).count());

        // Cleanup this thread
        free(scratch_cluster.N);
        free(scratch_cluster.pi);
        free(scratch_cluster.constant);
        free(scratch_cluster.avgvar);
        free(scratch_cluster.means);
        free(scratch_cluster.R);
        free(scratch_cluster.Rinv);
        free(scratch_cluster.memberships);
             
        CUDA_SAFE_CALL(cudaFree(um_fcs_data_by_event));
        CUDA_SAFE_CALL(cudaFree(um_fcs_data_by_dimension));

        CUDA_SAFE_CALL(cudaFree(um_clusters[tid].N));
        CUDA_SAFE_CALL(cudaFree(um_clusters[tid].pi));
        CUDA_SAFE_CALL(cudaFree(um_clusters[tid].constant));
        CUDA_SAFE_CALL(cudaFree(um_clusters[tid].avgvar));
        CUDA_SAFE_CALL(cudaFree(um_clusters[tid].means));
        CUDA_SAFE_CALL(cudaFree(um_clusters[tid].R));
        CUDA_SAFE_CALL(cudaFree(um_clusters[tid].Rinv));
        CUDA_SAFE_CALL(cudaFree(um_clusters[tid].memberships));
    } // end of parallel block

    // Main thread cleanup
    free(fcs_data_by_dimension);

    CUDA_SAFE_CALL(cudaFree(shared_likelihoods));
    CUDA_SAFE_CALL(cudaFree(um_clusters));

	*final_num_clusters = ideal_num_clusters;
	return saved_clusters;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) {
    int original_num_clusters, desired_num_clusters, ideal_num_clusters;
    
    // Validate the command-line arguments, parse # of clusters, etc 
    int error = validateArguments(argc,argv,&original_num_clusters,&desired_num_clusters);
    
    // Don't continue if we had a problem with the program arguments
    if(error) {
        return 1;
    }

    int num_dimensions;
    int num_events;
    
    // Read FCS data   
    PRINT("Parsing input file...");
    // This stores the data in a 1-D array with consecutive values being the dimensions from a single event
    // (num_events by num_dimensions matrix)
    float* fcs_data_by_event = readData(argv[2],&num_dimensions,&num_events);   

    if(!fcs_data_by_event) {
        printf("Error parsing input file. This could be due to an empty file ");
        printf("or an inconsistent number of dimensions. Aborting.\n");
        return 1;
    }
       
	clusters_t* clusters = cluster(original_num_clusters, desired_num_clusters, &ideal_num_clusters, num_dimensions, num_events, fcs_data_by_event);

	clusters_t saved_clusters;
	memcpy(&saved_clusters,clusters,sizeof(clusters_t));
 
    char const * result_suffix = ".results";
    char const* summary_suffix = ".summary";
    int filenamesize1 = strlen(argv[3]) + strlen(result_suffix) + 1;
    int filenamesize2 = strlen(argv[3]) + strlen(summary_suffix) + 1;
    char* result_filename = (char*) malloc(filenamesize1);
    char* summary_filename = (char*) malloc(filenamesize2);
    strcpy(result_filename,argv[3]);
    strcpy(summary_filename,argv[3]);
    strcat(result_filename,result_suffix);
    strcat(summary_filename,summary_suffix);

    
    PRINT("Summary filename: %s\n",summary_filename);
    PRINT("Results filename: %s\n",result_filename);
    
    // Open up the output file for cluster summary
    FILE* outf = fopen(summary_filename,"w");
    if(!outf) {
        printf("ERROR: Unable to open file '%s' for writing.\n",argv[3]);
        return -1;
    }

    // Print the clusters with the lowest rissanen score to the console and output file
    for(int c=0; c<ideal_num_clusters; c++) {
        if(ENABLE_PRINT) {
            // Output the final cluster stats to the console
            PRINT("Cluster #%d\n",c);
            printCluster(saved_clusters,c,num_dimensions);
            PRINT("\n\n");
        }

        if(ENABLE_OUTPUT) {
            // Output the final cluster stats to the output file        
            fprintf(outf,"Cluster #%d\n",c);
            writeCluster(outf,saved_clusters,c,num_dimensions);
            fprintf(outf,"\n\n");
        }
    }
    fclose(outf);
   
    if(ENABLE_OUTPUT) { 
        // Open another output file for the event level clustering results
        FILE* fresults = fopen(result_filename,"w");
        
        char header[1000];
        FILE* input_file = fopen(argv[2],"r");
        fgets(header,1000,input_file);
        fclose(input_file);
        fprintf(fresults,"%s",header);
        
        for(int i=0; i<num_events; i++) {
            for(int d=0; d<num_dimensions-1; d++) {
                fprintf(fresults,"%f,",fcs_data_by_event[i*num_dimensions+d]);
            }
            fprintf(fresults,"%f",fcs_data_by_event[i*num_dimensions+num_dimensions-1]);
            fprintf(fresults,"\t");
            for(int c=0; c<ideal_num_clusters-1; c++) {
                fprintf(fresults,"%f,",saved_clusters.memberships[c*num_events+i]);
            }
            fprintf(fresults,"%f",saved_clusters.memberships[(ideal_num_clusters-1)*num_events+i]);
            fprintf(fresults,"\n");
        }
        fclose(fresults);
    }
    
    // Cleanup host memory
    free(fcs_data_by_event);

    free(saved_clusters.N);
    free(saved_clusters.pi);
    free(saved_clusters.constant);
    free(saved_clusters.avgvar);
    free(saved_clusters.means);
    free(saved_clusters.R);
    free(saved_clusters.Rinv);
    free(saved_clusters.memberships);

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
// Validate command line arguments
///////////////////////////////////////////////////////////////////////////////
int validateArguments(int argc, char** argv, int* num_clusters, int* target_num_clusters) {
    if(argc <= 5 && argc >= 4) {
        // Parse num_clusters
        if(!sscanf(argv[1],"%d",num_clusters)) {
            printf("Invalid number of starting clusters\n\n");
            printUsage(argv);
            return 1;
        } 
        
        // Check bounds for num_clusters
        if(*num_clusters < 1) {
            printf("Invalid number of starting clusters\n\n");
            printUsage(argv);
            return 1;
        }
        
        // Parse infile
        FILE* infile = fopen(argv[2],"r");
        if(!infile) {
            printf("Invalid infile.\n\n");
            printUsage(argv);
            return 2;
        } 
        
        // Parse target_num_clusters
        if(argc == 5) {
            if(!sscanf(argv[4],"%d",target_num_clusters)) {
                printf("Invalid number of desired clusters.\n\n");
                printUsage(argv);
                return 4;
            }
            if(*target_num_clusters > *num_clusters) {
                printf("target_num_clusters must be less than equal to num_clusters\n\n");
                printUsage(argv);
                return 4;
            }
        } else {
            *target_num_clusters = 0;
        }
        
        // Clean up so the EPA is happy
        fclose(infile);
        return 0;
    } else {
        printUsage(argv);
        return 1;
    }
}

///////////////////////////////////////////////////////////////////////////////
// Print usage statement
///////////////////////////////////////////////////////////////////////////////
void printUsage(char** argv)
{
   printf("Usage: %s num_clusters infile outfile [target_num_clusters]\n",argv[0]);
   printf("\t num_clusters: The number of starting clusters\n");
   printf("\t infile: ASCII space-delimited FCS data file\n");
   printf("\t outfile: Clustering results output file\n");
   printf("\t target_num_clusters: A desired number of clusters. Must be less than or equal to num_clusters\n");
}
/*
 *print the values contained in clusters on a file
 */
void writeCluster(FILE* f, clusters_t clusters, int c, int num_dimensions) {
    fprintf(f,"Probability: %f\n", clusters.pi[c]);
    fprintf(f,"N: %f\n",clusters.N[c]);
    fprintf(f,"Means: ");
    for(int i=0; i<num_dimensions; i++){
        fprintf(f,"%f ",clusters.means[c*num_dimensions+i]);
    }
    fprintf(f,"\n");

    fprintf(f,"\nR Matrix:\n");
    for(int i=0; i<num_dimensions; i++) {
        for(int j=0; j<num_dimensions; j++) {
            fprintf(f,"%f ", clusters.R[c*num_dimensions*num_dimensions+i*num_dimensions+j]);
        }
        fprintf(f,"\n");
    }
    fflush(f);   
    /*
    fprintf(f,"\nR-inverse Matrix:\n");
    for(int i=0; i<num_dimensions; i++) {
        for(int j=0; j<num_dimensions; j++) {
            fprintf(f,"%.3f ", c->Rinv[i*num_dimensions+j]);
        }
        fprintf(f,"\n");
    } 
    */
}
/*
 * print the values contained in clusters on stdout
 */
void printCluster(clusters_t clusters, int c, int num_dimensions) {
    writeCluster(stdout,clusters,c,num_dimensions);
}
/*
 *calculate the distance between two clusters
 */
float cluster_distance(clusters_t clusters, int c1, int c2, clusters_t temp_cluster, int num_dimensions) {
    // Add the clusters together, this updates pi,means,R,N and stores in temp_cluster
    add_clusters(clusters,c1,c2,temp_cluster,num_dimensions);
    
    return clusters.N[c1]*clusters.constant[c1] + clusters.N[c2]*clusters.constant[c2] - temp_cluster.N[0]*temp_cluster.constant[0];
}
/*
 *add the clusters together, stores the result in temp_cluster
 */
void add_clusters(clusters_t clusters, int c1, int c2, clusters_t temp_cluster, int num_dimensions) {
    float wt1,wt2;
 
    wt1 = (clusters.N[c1]) / (clusters.N[c1] + clusters.N[c2]);
    wt2 = 1.0f - wt1;
    
    // Compute new weighted means
    for(int i=0; i<num_dimensions;i++) {
        temp_cluster.means[i] = wt1*clusters.means[c1*num_dimensions+i] + wt2*clusters.means[c2*num_dimensions+i];
    }
    
    // Compute new weighted covariance
    for(int i=0; i<num_dimensions; i++) {
        for(int j=i; j<num_dimensions; j++) {
            // Compute R contribution from cluster1
            temp_cluster.R[i*num_dimensions+j] = ((temp_cluster.means[i]-clusters.means[c1*num_dimensions+i])
                                                *(temp_cluster.means[j]-clusters.means[c1*num_dimensions+j])
                                                +clusters.R[c1*num_dimensions*num_dimensions+i*num_dimensions+j])*wt1;
            // Add R contribution from cluster2
            temp_cluster.R[i*num_dimensions+j] += ((temp_cluster.means[i]-clusters.means[c2*num_dimensions+i])
                                                    *(temp_cluster.means[j]-clusters.means[c2*num_dimensions+j])
                                                    +clusters.R[c2*num_dimensions*num_dimensions+i*num_dimensions+j])*wt2;
            // Because its symmetric...
            temp_cluster.R[j*num_dimensions+i] = temp_cluster.R[i*num_dimensions+j];
        }
    }
    
    // Compute pi
    temp_cluster.pi[0] = clusters.pi[c1] + clusters.pi[c2];
    
    // Compute N
    temp_cluster.N[0] = clusters.N[c1] + clusters.N[c2];

    float log_determinant;
    // Copy R to Rinv matrix
    memcpy(temp_cluster.Rinv,temp_cluster.R,sizeof(float)*num_dimensions*num_dimensions);
    // Invert the matrix using LU decomposition with partial pivoting.
    invert_cpu(temp_cluster.Rinv,num_dimensions,&log_determinant);
    // Compute the constant
    temp_cluster.constant[0] = (-num_dimensions)*0.5f*logf(2.0f*PI)-0.5f*log_determinant;
    
    // avgvar same for all clusters
    temp_cluster.avgvar[0] = clusters.avgvar[0];
}
/*
 * copy the value contained in src(source) in dest(destination)
 */
void copy_cluster(clusters_t dest, int c_dest, clusters_t src, int c_src, int num_dimensions) {
    dest.N[c_dest] = src.N[c_src];
    dest.pi[c_dest] = src.pi[c_src];
    dest.constant[c_dest] = src.constant[c_src];
    dest.avgvar[c_dest] = src.avgvar[c_src];
    memcpy(&(dest.means[c_dest*num_dimensions]),&(src.means[c_src*num_dimensions]),sizeof(float)*num_dimensions);
    memcpy(&(dest.R[c_dest*num_dimensions*num_dimensions]),&(src.R[c_src*num_dimensions*num_dimensions]),sizeof(float)*num_dimensions*num_dimensions);
    memcpy(&(dest.Rinv[c_dest*num_dimensions*num_dimensions]),&(src.Rinv[c_src*num_dimensions*num_dimensions]),sizeof(float)*num_dimensions*num_dimensions);
}