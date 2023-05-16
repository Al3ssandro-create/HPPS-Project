import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.stream.JsonReader;

import java.io.*;
import java.util.ArrayList;
import java.util.Scanner;

public class GMM {
    private Context context;

    private int TRUNCATE = 1;
    private int PRINT = 1;
    private int DEBUG = 1;
    private int NUM_BLOCKS = 24;
    private int NUM_CLUSTERS_PER_BLOCK = 6;
    private int NUM_THREADS_MSTEP = 256;

    private int num_gpus;
    private int num_dimensions;
    private int num_events;
    private int ideal_num_clusters;
    private int final_num_clusters;

    private Value seed_clusters;
    private Value constants_kernel;
    private Value estep1;
    private Value estep2;
    private Value mstep_N;
    private Value mstep_means;
    private Value mstep_covariance2;

    public GMM(Config config) {
        this.context = createContext(config);
        this.num_gpus = config.num_gpus;

        // Building the kernel in order to be able to call them later
        Value buildKernel = this.context.eval("grcuda", "buildkernel");
        seed_clusters = buildKernel.execute(GMMKernels.seed_clusters, "seed_clusters",
                "pointer, pointer, pointer, pointer, pointer, pointer, sint32, sint32, sint32");
        constants_kernel = buildKernel.execute(GMMKernels.constants_kernel, "constants_kernel",
                "pointer, pointer, pointer, pointer, pointer, sint32, sint32");
        estep1 = buildKernel.execute(GMMKernels.estep1, "estep1",
                "pointer, pointer, pointer, pointer, pointer, pointer, sint32, sint32");
        estep2 = buildKernel.execute(GMMKernels.estep2, "estep2",
                "pointer, sint32, sint32, sint32, pointer");
        mstep_N = buildKernel.execute(GMMKernels.mstep_N, "mstep_N",
                "pointer, pointer, pointer, sint32, sint32, sint32");
        mstep_means = buildKernel.execute(GMMKernels.mstep_means, "mstep_means",
                "pointer, pointer, pointer, sint32, sint32, sint32");
        mstep_covariance2 = buildKernel.execute(GMMKernels.mstep_covariance2, "mstep_covariance2",
                "pointer, pointer, pointer, pointer, pointer, sint32, sint32, sint32");
    }

    private float[] readData(String fileName) {
        int length = fileName.length();
        System.out.println("File Extension: " + fileName.substring(length - 3));
        if (fileName.endsWith("bin")) {
            return readBIN(fileName);
        } else {
            return readCSV(fileName);
        }
    }

    private float[] readBIN(String fileName) {
        // TODO
        return null;
    }

    private float[] readCSV(String fileName) {
        int num_dims = 0;
        String line1;
        ArrayList<String> lines = new ArrayList<>();

        try (Scanner scanner = new Scanner(new File(fileName))) {
            while (scanner.hasNextLine()) {
                lines.add(scanner.nextLine());
            }
        } catch (FileNotFoundException e) {
            System.out.println("Unable to read the file " + fileName);
            return null;
        }

        if (lines.size() > 0) {
            line1 = lines.get(0);
            String line2 = line1;

            // TODO: Why is he used these 'tokens'
            // START
            String[] temp;
            temp = line1.split(",");

            num_dims = temp.length;
            // END

            lines.remove(0); // Remove first line, assumed to be header
            int num_events = lines.size();

            if (this.TRUNCATE == 1) {
                System.out.println("Number of events removed to ensure memory alignment " + num_events % (16 * 2));
                num_events -= num_events % (16 * 2);
            }

            // Allocate space for all the FCS data
            float[] data = new float[num_dims * num_events];

            for (int i = 0; i < num_events; i++) {
                temp = lines.get(i).split(",");

                for (int j = 0; j < num_dims; j++) {
                    if (temp.length < num_dims) {
                        return null;
                    }
                    data[i * num_dims + j] = Float.parseFloat(temp[j].split(" ")[0]);
                }
            }

            this.num_dimensions = num_dims;
            this.num_events = num_events;

            return data;
        } else {
            return null;
        }
    }

    private cluster_t cluster(int original_num_clusters, int desired_num_clusters, float[] fcs_data_by_event) {
        int regroup_iterations = 0;
        int params_iterations = 0;
        int constants_iterations = 0;
        int reduce_iterations = 0;
        int ideal_num_clusters;
        int stop_number;

        // Number of clusters to stop iterating at.
        if (desired_num_clusters == 0) {
            stop_number = 1;
        } else {
            stop_number = desired_num_clusters;
        }

        if (num_gpus < 1) {
            System.out.println("ERROR: No CUDA capable GPUs detected.");
            return null;
        } else if (num_gpus == 1) {
            System.out.println("Warning: Only 1 CUDA GPU detected. Running single GPU version would be more efficient.");
        } else {
            if (PRINT == 1) System.out.println("Using " + num_gpus + " GPUs");
        }

        // Transpose the event data (allows coalesced access pattern in E-step kernel)
        // This has consecutive values being from the same dimension of the data
        // (num_dimensions by num_events matrix)
        float[] fcs_data_by_dimension = new float[num_events * num_dimensions];

        for (int e = 0; e < num_events; e++) {
            for (int d = 0; d < num_dimensions; d++) {
                fcs_data_by_dimension[d * num_events + e] = fcs_data_by_event[e * num_dimensions + d];
            }
        }

        if (PRINT == 1) System.out.println("Number of events: " + num_events);
        if (PRINT == 1) System.out.println("Number of dimensions: " + num_dimensions);
        if (PRINT == 1) System.out.println("Starting with " + original_num_clusters + "cluster(s), will stop at " + stop_number + " cluster(s).");

        // Setup the cluster data structures on host
        // This the shared memory space between the GPUs
        clusters_um[] um_clusters = new clusters_um[num_gpus];

        // Only need one copy of all the memberships
        float[] special_memberships = new float[num_events * original_num_clusters];

        // Declare another set of clusters for saving the results of the best configuration
        clusters_t saved_clusters = new cluster_t();
        saved_clusters.N = new float[original_num_clusters];
        saved_clusters.pi = new float[original_num_clusters];
        saved_clusters.constant = new float[original_num_clusters];
        saved_clusters.avgvar = new float[original_num_clusters];
        saved_clusters.means = new float[num_dimensions * original_num_clusters];
        saved_clusters.R = new float[num_dimensions * num_dimensions * original_num_clusters];
        saved_clusters.Rinv = new float[num_dimensions * num_dimensions * original_num_clusters];
        saved_clusters.memberships = new float[num_events * original_num_clusters];

        if (DEBUG == 1) System.out.println("Finished allocating shared cluster structures on host");

        // Used to hold the result from regroup kernel
        Value[] shared_likelihoods = new Value[NUM_BLOCKS * num_gpus];
        float likelihood, old_likelihood;
        float min_rissanen;

        // ------ OPENMP START ------

        // TODO: check here the size, they are different from 'saved_clusters'
        clusters_t scratch_cluster = new cluster_t();
        scratch_cluster.N = new float[1];
        scratch_cluster.pi = new float[1];
        scratch_cluster.constant = new float[1];
        scratch_cluster.avgvar = new float[1];
        scratch_cluster.means = new float[num_dimensions];
        scratch_cluster.R = new float[num_dimensions * num_dimensions];
        scratch_cluster.Rinv = new float[num_dimensions * num_dimensions];
        scratch_cluster.memberships = new float[num_events];

        if (DEBUG == 1) System.out.println("Finished allocating memory on host for clusters.");

        // determine how many events this gpu will handle
        int events_per_gpu = num_events / num_gpus;
        int[] my_num_events = new int[num_gpus];
        for (int i = 0; i < num_gpus; i++) {
            my_num_events[i] = events_per_gpu;
            if (i == num_gpus - 1)
                my_num_events[i] += num_events % num_gpus; // last gpu has to handle the remaining uneven events
            if (DEBUG == 1) System.out.println("GPU " + i + " will handle " + my_num_events + "events");
        }

        // Setup the cluster data structures on device
        // First allocate structures on the host, CUDA malloc the arrays
        // Then CUDA malloc structures on the device and copy them over
        for (int i = 0; i < num_gpus; i++) {
            um_clusters[i].N = context.eval("grcuda", "float[" + original_num_clusters + "]");
            um_clusters[i].pi = context.eval("grcuda", "float[" + original_num_clusters + "]");
            um_clusters[i].constant = context.eval("grcuda", "float[" + original_num_clusters + "]");
            um_clusters[i].avgvar = context.eval("grcuda", "float[" + original_num_clusters + "]");
            um_clusters[i].means = context.eval("grcuda", "float[" + num_dimensions * original_num_clusters + "]");
            um_clusters[i].R = context.eval("grcuda", "float[" + num_dimensions * num_dimensions * original_num_clusters + "]");
            um_clusters[i].Rinv = context.eval("grcuda", "float[" + num_dimensions * num_dimensions * original_num_clusters + "]");
            um_clusters[i].memberships = context.eval("grcuda", "float[" + my_num_events[i] * (original_num_clusters + NUM_CLUSTERS_PER_BLOCK - original_num_clusters % NUM_CLUSTERS_PER_BLOCK) + "]");
        }

        // Allocate a struct on the device
        if (DEBUG == 1) System.out.println("Finished allocating memory on device for clusters.");

        // Allocate device memory for FCS data and copy relavant FCS data to device.
        int[] mem_size = new int[num_gpus];
        for (int i = 0; i < num_gpus; i++) mem_size[i] = num_dimensions * my_num_events[i];

        Value[] um_fcs_data_by_event = new Value[num_gpus];
        Value[] um_fcs_data_by_dimension = new Value[num_gpus];
        for (int i = 0; i < num_gpus; i++) {
            um_fcs_data_by_event[i] = context.eval("grcuda", "float[" + mem_size[i] + "]");
            um_fcs_data_by_dimension[i] = context.eval("grcuda", "float[" + mem_size[i] + "]");
        }

        for (int i = 0; i<num_gpus; i++) {
            //memcpy(um_fcs_data_by_event, &fcs_data_by_event[num_dimensions*events_per_gpu*i], mem_size[i]);
            memcpy();
        }

        // Copying the transposed data is trickier since it's not all contigious for the relavant events
        for (int i = 0; i < num_gpus; i++) {
            for (int d = 0; d < num_dimensions; d++) {
                //memcpy( &um_fcs_data_by_dimension[d * my_num_events], &fcs_data_by_dimension[d * num_events + tid * events_per_gpu], sizeof( float)*my_num_events);
                memcpy();
            }
            if (DEBUG == 1) System.out.println("GPU " + i + ": Finished copying FCS data to device.");
        }

        //////////////// Initialization done, starting kernels ////////////////
        if (DEBUG == 1) System.out.println("Invoking seed_clusters kernel.");

        // seed_clusters sets initial pi values,
        // finds the means / covariances and copies it to all the clusters

        // ------ OPENMP MASTER START (NUM_GPU == 0) ------
        seed_clusters.execute(1, NUM_THREADS_MSTEP)
                .execute(um_fcs_data_by_event[0], um_clusters[0].means, um_clusters[0].R,
                        um_clusters[0].N, um_clusters[0].pi, um_clusters[0].avgvar, num_dimensions, original_num_clusters, my_num_events[0]);

        if (DEBUG == 1) System.out.println("Invoking constants kernel.");
            // Computes the R matrix inverses, and the gaussian constant
        constants_kernel.execute(original_num_clusters, NUM_THREADS_MSTEP)
                .execute(um_clusters[0].R, um_clusters[0].Rinv, um_clusters[0].constant, um_clusters[0].pi, original_num_clusters, num_dimensions);

        constants_iterations++;

        //seed_clusters( & um_clusters[0], fcs_data_by_event, original_num_clusters, num_dimensions, num_events);
        seed_clusters();

        if (DEBUG == 1) System.out.println("Starting Clusters");
        for (int c = 0; c < original_num_clusters; c++) {
            if (DEBUG == 1) System.out.println("Cluster #" + c);

            if (DEBUG == 1) System.out.println("\tN: " + um_clusters[0].N[c]);

            if (DEBUG == 1) System.out.println("\tpi: " + um_clusters[0].pi[c]);

            if (DEBUG == 1) System.out.println("\tMeans:");
            for (int d = 0; d < num_dimensions; d++)
                if (DEBUG == 1) System.out.println("\t\t" + um_clusters[0].means[c * num_dimensions + d]);

            if (DEBUG == 1) System.out.println("\tR:");
            for (int d = 0; d < num_dimensions; d++)
                for (int e = 0; e < num_dimensions; e++)
                    if (DEBUG == 1) System.out.println("\t\t" + um_clusters[0].R[c * num_dimensions * num_dimensions + d * num_dimensions + e]);

            if (DEBUG == 1) System.out.println("\tR-inverse:");
            for (int d = 0; d < num_dimensions; d++)
                for (int e = 0; e < num_dimensions; e++)
                    if (DEBUG == 1) System.out.println("\t\t" + um_clusters[0].Rinv[c * num_dimensions * num_dimensions + d * num_dimensions + e]);

            DEBUG("\tAvgvar: " + um_clusters[0].avgvar[c]);

            DEBUG("\tConstant: " + um_clusters[0].constant[c]);
        }
        // ------ OPENMP MASTER END (NUM_GPU == 0)------

        // TODO: TO BE CONTINUED

        // synchronize after first gpu does the seeding, copy result to all gpus
        #pragma omp barrier
        startTimer(timers.memcpy);
        memcpy(um_clusters[tid].N, um_clusters[0].N, sizeof( float)*original_num_clusters);
        memcpy(um_clusters[tid].pi, um_clusters[0].pi, sizeof( float)*original_num_clusters);
        memcpy(um_clusters[tid].constant, um_clusters[0].constant, sizeof( float)*original_num_clusters);
        memcpy(um_clusters[tid].avgvar, um_clusters[0].avgvar, sizeof( float)*original_num_clusters);
        memcpy(um_clusters[tid].means, um_clusters[0].means, sizeof( float)*num_dimensions * original_num_clusters);
        memcpy(um_clusters[tid].R, um_clusters[0].R, sizeof( float)*
        num_dimensions * num_dimensions * original_num_clusters);
        memcpy(um_clusters[tid].Rinv, um_clusters[0].Rinv, sizeof( float)*
        num_dimensions * num_dimensions * original_num_clusters);
        stopTimer(timers.memcpy);

        startTimer(timers.cpu);
        // Calculate an epsilon value
        //int ndata_points = num_events*num_dimensions;
        float epsilon = (1 + num_dimensions + 0.5f * (num_dimensions + 1) * num_dimensions) * logf((float) num_events * num_dimensions) * 0.001f;
        int iters;

        //epsilon = 1e-6;
        #pragma omp master
        PRINT("Gaussian.cu: epsilon = %f\n", epsilon);

        // Variables for GMM reduce order
        float distance, min_distance = 0.0;
        float rissanen;
        int min_c1, min_c2;
        stopTimer(timers.cpu);

        #pragma omp barrier
        auto start1 = std::chrono::steady_clock::now ();

        for (int num_clusters = original_num_clusters; num_clusters >= stop_number; num_clusters--) {
            /*************** EM ALGORITHM *****************************/

            // do initial E-step
            // Calculates a cluster membership probability
            // for each event and each cluster.
            DEBUG("Invoking E-step kernels.");
            startTimer(timers.e_step);
            estep1 << < dim3(num_clusters, NUM_BLOCKS), NUM_THREADS_ESTEP >>> (um_fcs_data_by_dimension, &
            (um_clusters[tid]), num_dimensions, my_num_events);
            estep2 << < NUM_BLOCKS, NUM_THREADS_ESTEP >>> (um_fcs_data_by_dimension, &
            (um_clusters[tid]), num_dimensions, num_clusters, my_num_events, &shared_likelihoods[tid * NUM_BLOCKS]);
            cudaDeviceSynchronize();
            #pragma omp master
            {
                regroup_iterations++;
            }
            // check if kernel execution generated an error
            //CUT_CHECK_ERROR("Kernel execution failed");
            stopTimer(timers.e_step);

            #pragma omp barrier
            startTimer(timers.cpu);

            #pragma omp master
            {
                likelihood = 0.0;
                for (int i = 0; i < NUM_BLOCKS * num_gpus; i++) {
                    likelihood += shared_likelihoods[i];
                }
                DEBUG("Likelihood: %e\n", likelihood);
            }
            #pragma omp barrier
            stopTimer(timers.cpu);

            float change = epsilon * 2;

            #pragma omp master
            PRINT("Performing EM algorithm on %d clusters.\n", num_clusters);
            iters = 0;
            // This is the iterative loop for the EM algorithm.
            // It re-estimates parameters, re-computes constants, and then regroups the events
            // These steps keep repeating until the change in likelihood is less than some epsilon
            while (iters < MIN_ITERS || (fabs(change) > epsilon && iters < MAX_ITERS)) {
                #pragma omp master
                {
                    old_likelihood = likelihood;
                }

                DEBUG("Invoking reestimate_parameters (M-step) kernel.");
                startTimer(timers.m_step);
                // This kernel computes a new N, pi isn't updated until compute_constants though
                mstep_N << < num_clusters, NUM_THREADS_MSTEP >>> ( & (um_clusters[tid]), num_dimensions, num_clusters, my_num_events)
                ;
                cudaDeviceSynchronize();
                stopTimer(timers.m_step);

                // TODO: figure out the omp reduction pragma...
                // Reduce N for all clusters, copy back to device
                #pragma omp barrier
                startTimer(timers.cpu);
                #pragma omp master
                {
                    for (int g = 1; g < num_gpus; g++) {
                        for (int c = 0; c < num_clusters; c++) {
                            um_clusters[0].N[c] += um_clusters[g].N[c];
                            DEBUG("Cluster %d: N = %f\n", c, um_clusters[0].N[c]);
                        }
                    }
                }
                #pragma omp barrier
                stopTimer(timers.cpu);
                startTimer(timers.memcpy);
                memcpy(um_clusters[tid].N, um_clusters[0].N, sizeof( float)*num_clusters);
                stopTimer(timers.memcpy);

                startTimer(timers.m_step);
                dim3 gridDim1 (num_clusters, num_dimensions);
                mstep_means << < gridDim1, NUM_THREADS_MSTEP >>> (um_fcs_data_by_dimension, &
                (um_clusters[tid]), num_dimensions, num_clusters, my_num_events);
                cudaDeviceSynchronize();
                stopTimer(timers.m_step);

                // Reduce means for all clusters, copy back to device
                #pragma omp barrier
                startTimer(timers.cpu);
                #pragma omp master
                {
                    for (int g = 1; g < num_gpus; g++) {
                        for (int c = 0; c < num_clusters; c++) {
                            for (int d = 0; d < num_dimensions; d++) {
                                um_clusters[0].means[c * num_dimensions + d] += um_clusters[g].means[c * num_dimensions + d];
                            }
                        }
                    }
                    for (int c = 0; c < num_clusters; c++) {
                        DEBUG("Cluster %d  Means:", c, um_clusters[0].N[c]);
                        for (int d = 0; d < num_dimensions; d++) {
                            if (um_clusters[0].N[c] > 0.5f) {
                                um_clusters[0].means[c * num_dimensions + d] /= um_clusters[0].N[c];
                            } else {
                                um_clusters[0].means[c * num_dimensions + d] = 0.0f;
                            }
                            DEBUG(" %f", um_clusters[0].means[c * num_dimensions + d]);
                        }
                        DEBUG("\n");
                    }
                }
                #pragma omp barrier
                stopTimer(timers.cpu);
                startTimer(timers.memcpy);
                memcpy(um_clusters[tid].means, um_clusters[0].means, sizeof( float)*num_clusters * num_dimensions);
                stopTimer(timers.memcpy);

                startTimer(timers.m_step);
                // Covariance is symmetric, so we only need to compute N*(N+1)/2 matrix elements per cluster
                dim3 gridDim2 (num_clusters, num_dimensions * (num_dimensions + 1) / 2);
                //mstep_covariance1<<<gridDim2, NUM_THREADS_MSTEP>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,my_num_events);
                mstep_covariance2 << < dim3((num_clusters + NUM_CLUSTERS_PER_BLOCK - 1) / NUM_CLUSTERS_PER_BLOCK, num_dimensions * (num_dimensions + 1) / 2), NUM_THREADS_MSTEP >>> (um_fcs_data_by_dimension, &
                (um_clusters[tid]), num_dimensions, num_clusters, my_num_events);
                cudaDeviceSynchronize();
                stopTimer(timers.m_step);

                // Reduce R for all clusters, copy back to device
                #pragma omp barrier
                startTimer(timers.cpu);
                #pragma omp master
                {
                    for (int g = 1; g < num_gpus; g++) {
                        for (int c = 0; c < num_clusters; c++) {
                            for (int d = 0; d < num_dimensions * num_dimensions; d++) {
                                um_clusters[0].R[c * num_dimensions * num_dimensions + d] += um_clusters[g].R[c * num_dimensions * num_dimensions + d];
                            }
                        }
                    }
                    for (int c = 0; c < num_clusters; c++) {
                        if (um_clusters[0].N[c] > 0.5f) {
                            for (int d = 0; d < num_dimensions * num_dimensions; d++) {
                                um_clusters[0].R[c * num_dimensions * num_dimensions + d] /= um_clusters[0].N[c];
                            }
                        } else {
                            for (int i = 0; i < num_dimensions; i++) {
                                for (int j = 0; j < num_dimensions; j++) {
                                    if (i == j) {
                                        um_clusters[0].R[c * num_dimensions * num_dimensions + i * num_dimensions + j] = 1.0;
                                    } else {
                                        um_clusters[0].R[c * num_dimensions * num_dimensions + i * num_dimensions + j] = 0.0;
                                    }
                                }
                            }
                        }
                    }
                }
                #pragma omp barrier
                stopTimer(timers.cpu);
                startTimer(timers.memcpy);
                memcpy(um_clusters[tid].R, um_clusters[0].R, sizeof( float)*
                num_clusters * num_dimensions * num_dimensions);
                stopTimer(timers.memcpy);

                cudaDeviceSynchronize();
                //CUT_CHECK_ERROR("M-step Kernel execution failed: ");
                #pragma omp master
                {
                    params_iterations++;
                }

                DEBUG("Invoking constants kernel.");
                // Inverts the R matrices, computes the constant, normalizes cluster probabilities
                startTimer(timers.constants);
                constants_kernel << < num_clusters, NUM_THREADS_MSTEP >>> ( & (um_clusters[tid]), num_clusters, num_dimensions)
                ;
                cudaDeviceSynchronize();

                #pragma omp master
                {
                    for (int temp_c = 0; temp_c < num_clusters; temp_c++)
                        DEBUG("Cluster %d constant: %e\n", temp_c, um_clusters[tid].constant[temp_c]);
                }
                stopTimer(timers.constants);
                //CUT_CHECK_ERROR("Constants Kernel execution failed: ");
                #pragma omp master
                {
                    constants_iterations++;
                }

                DEBUG("Invoking regroup (E-step) kernel with %d blocks.\n", NUM_BLOCKS);
                startTimer(timers.e_step);
                // Compute new cluster membership probabilities for all the events
                estep1 << < dim3(num_clusters, NUM_BLOCKS), NUM_THREADS_ESTEP >>> (um_fcs_data_by_dimension, &
                (um_clusters[tid]), num_dimensions, my_num_events);
                estep2 << < NUM_BLOCKS, NUM_THREADS_ESTEP >>> (um_fcs_data_by_dimension, &
                (um_clusters[tid]), num_dimensions, num_clusters, my_num_events, &shared_likelihoods[tid * NUM_BLOCKS]);
                cudaDeviceSynchronize();
                //CUT_CHECK_ERROR("E-step Kernel execution failed: ");
                stopTimer(timers.e_step);
                #pragma omp master
                {
                    regroup_iterations++;
                }

                // check if kernel execution generated an error
                //CUT_CHECK_ERROR("Kernel execution failed");

                #pragma omp barrier
                startTimer(timers.cpu);
                #pragma omp master
                {
                    likelihood = 0.0;
                    for (int i = 0; i < NUM_BLOCKS * num_gpus; i++) {
                        likelihood += shared_likelihoods[i];
                    }
                    DEBUG("Likelihood: %e\n", likelihood);
                }
                stopTimer(timers.cpu);
                #pragma omp barrier // synchronize for likelihood

                        change = likelihood - old_likelihood;
                DEBUG("GPU %d, Change in likelihood: %e\n", tid, change);

                iters++;
                #pragma omp barrier // synchronize loop iteration
            }

            DEBUG("GPU %d done with EM loop\n", tid);

            startTimer(timers.memcpy);

            stopTimer(timers.memcpy);
            startTimer(timers.cpu);
            for (int c = 0; c < num_clusters; c++) {
                memcpy( & (special_memberships[c * num_events + tid * events_per_gpu]), &
                (um_clusters[tid].memberships[c * my_num_events]), sizeof( float)*my_num_events);
            }
            #pragma omp barrier
            DEBUG("GPU %d done with copying cluster data from device\n", tid);

            // Calculate Rissanen Score
            rissanen = -likelihood + 0.5f * (num_clusters * (1.0f + num_dimensions + 0.5f * (num_dimensions + 1.0f) * num_dimensions) - 1.0f) * logf((float) num_events * num_dimensions);
            #pragma omp master
            PRINT("\nLikelihood: %e\n", likelihood);
            #pragma omp master
            PRINT("\nRissanen Score: %e\n", rissanen);

            #pragma omp barrier
            #pragma omp master
            {
                // Save the cluster data the first time through, so we have a base rissanen score and result
                // Save the cluster data if the solution is better and the user didn't specify a desired number
                // If the num_clusters equals the desired number, stop
                if (num_clusters == original_num_clusters || (rissanen < min_rissanen && desired_num_clusters == 0) || (num_clusters == desired_num_clusters)) {
                    min_rissanen = rissanen;
                    ideal_num_clusters = num_clusters;
                    // Save the cluster configuration somewhere
                    memcpy(saved_clusters -> N, um_clusters[0].N, sizeof( float)*num_clusters);
                    memcpy(saved_clusters -> pi, um_clusters[0].pi, sizeof( float)*num_clusters);
                    memcpy(saved_clusters -> constant, um_clusters[0].constant, sizeof( float)*num_clusters);
                    memcpy(saved_clusters -> avgvar, um_clusters[0].avgvar, sizeof( float)*num_clusters);
                    memcpy(saved_clusters -> means, um_clusters[0].means, sizeof( float)*num_dimensions * num_clusters);
                    memcpy(saved_clusters -> R, um_clusters[0].R, sizeof( float)*
                    num_dimensions * num_dimensions * num_clusters);
                    memcpy(saved_clusters -> Rinv, um_clusters[0].Rinv, sizeof( float)*
                    num_dimensions * num_dimensions * num_clusters);
                    memcpy(saved_clusters -> memberships, special_memberships, sizeof( float)*num_events * num_clusters)
                    ;
                }
            }
            #pragma omp barrier
            stopTimer(timers.cpu);
            /**************** Reduce GMM Order ********************/
            startTimer(timers.reduce);
            // Don't want to reduce order on the last iteration
            if (num_clusters > stop_number) {
                startTimer(timers.cpu);
                #pragma omp master
                {
                    // First eliminate any "empty" clusters
                    for (int i = num_clusters - 1; i >= 0; i--) {
                        if (um_clusters[0].N[i] < 0.5) {
                            DEBUG("Cluster #%d has less than 1 data point in it.\n", i);
                            for (int j = i; j < num_clusters - 1; j++) {
                                copy_cluster(um_clusters[0], j, um_clusters[0], j + 1, num_dimensions);
                            }
                            num_clusters--;
                        }
                    }

                    min_c1 = 0;
                    min_c2 = 1;
                    DEBUG("Number of non-empty clusters: %d\n", num_clusters);
                    // For all combinations of subclasses...
                    // If the number of clusters got really big might need to do a non-exhaustive search
                    // Even with 100*99/2 combinations this doesn't seem to take too long
                    for (int c1 = 0; c1 < num_clusters; c1++) {
                        for (int c2 = c1 + 1; c2 < num_clusters; c2++) {
                            // compute distance function between the 2 clusters
                            distance = cluster_distance(um_clusters[0], c1, c2, scratch_cluster, num_dimensions);

                            // Keep track of minimum distance
                            if ((c1 == 0 && c2 == 1) || distance < min_distance) {
                                min_distance = distance;
                                min_c1 = c1;
                                min_c2 = c2;
                            }
                        }
                    }

                    PRINT("\nMinimum distance between (%d,%d). Combining clusters\n", min_c1, min_c2);
                    // Add the two clusters with min distance together
                    //add_clusters(&(clusters[min_c1]),&(clusters[min_c2]),scratch_cluster,num_dimensions);
                    add_clusters(um_clusters[0], min_c1, min_c2, scratch_cluster, num_dimensions);
                    // Copy new combined cluster into the main group of clusters, compact them
                    //copy_cluster(&(clusters[min_c1]),scratch_cluster,num_dimensions);
                    copy_cluster(um_clusters[0], min_c1, scratch_cluster, 0, num_dimensions);
                    for (int i = min_c2; i < num_clusters - 1; i++) {
                        //printf("Copying cluster %d to cluster %d\n",i+1,i);
                        //copy_cluster(&(clusters[i]),&(clusters[i+1]),num_dimensions);
                        copy_cluster(um_clusters[0], i, um_clusters[0], i + 1, num_dimensions);
                    }
                }
                stopTimer(timers.cpu);
                #pragma omp barrier

                startTimer(timers.memcpy);
                // Copy the clusters back to the device
                memcpy(um_clusters[tid].N, um_clusters[0].N, sizeof( float)*num_clusters);
                memcpy(um_clusters[tid].pi, um_clusters[0].pi, sizeof( float)*num_clusters);
                memcpy(um_clusters[tid].constant, um_clusters[0].constant, sizeof( float)*num_clusters);
                memcpy(um_clusters[tid].avgvar, um_clusters[0].avgvar, sizeof( float)*num_clusters);
                memcpy(um_clusters[tid].means, um_clusters[0].means, sizeof( float)*num_dimensions * num_clusters);
                memcpy(um_clusters[tid].R, um_clusters[0].R, sizeof( float)*
                num_dimensions * num_dimensions * num_clusters);
                memcpy(um_clusters[tid].Rinv, um_clusters[0].Rinv, sizeof( float)*
                num_dimensions * num_dimensions * num_clusters);
                stopTimer(timers.memcpy);

                startTimer(timers.cpu);
                for (int c = 0; c < num_clusters; c++) {
                    memcpy( & um_clusters[tid].memberships[c * my_num_events], &
                    (special_memberships[c * num_events + tid * (num_events / num_gpus)]), sizeof( float)*my_num_events)
                    ;
                }
                stopTimer(timers.cpu);
            } // GMM reduction block
            stopTimer(timers.reduce);
            #pragma omp master
            {
                reduce_iterations++;
            }

            #pragma omp barrier
        } // outer loop from M to 1 clusters
        #pragma omp master
        PRINT("\nFinal rissanen Score was: %f, with %d clusters.\n", min_rissanen, ideal_num_clusters);
        #pragma omp barrier

        auto diff1 = std::chrono::steady_clock::now () - start1;
        PROFILING("##### Time execution on device %d: %ld micro sec. #####\n", tid, std::chrono::duration_cast < std::chrono::
        microseconds > (diff1).count());

        // Print some profiling information
        printf("GPU %d:\n\tE-step Kernel:\t%7.4f\t%d\t%7.4f\n\tM-step Kernel:\t%7.4f\t%d\t%7.4f\n\tConsts Kernel:\t%7.4f\t%d\t%7.4f\n\tOrder Reduce:\t%7.4f\t%d\t%7.4f\n\tGPU Memcpy:\t%7.4f\n\tCPU:\t\t%7.4f\n", tid, getTimerValue(timers.e_step) / 1000.0, regroup_iterations, (double) getTimerValue(timers.e_step) / (double) regroup_iterations / 1000.0, getTimerValue(timers.m_step) / 1000.0, params_iterations, (double) getTimerValue(timers.m_step) / (double) params_iterations / 1000.0, getTimerValue(timers.constants) / 1000.0, constants_iterations, (double) getTimerValue(timers.constants) / (double) constants_iterations / 1000.0, getTimerValue(timers.reduce) / 1000.0, reduce_iterations, (double) getTimerValue(timers.reduce) / (double) reduce_iterations / 1000.0, getTimerValue(timers.memcpy) / 1000.0, getTimerValue(timers.cpu) / 1000.0);

        cleanup_profile_t( & timers);

        free(scratch_cluster.N);
        free(scratch_cluster.pi);
        free(scratch_cluster.constant);
        free(scratch_cluster.avgvar);
        free(scratch_cluster.means);
        free(scratch_cluster.R);
        free(scratch_cluster.Rinv);
        free(scratch_cluster.memberships);

        // cleanup GPU memory

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

    CUDA_SAFE_CALL(cudaFree(shared_likelihoods));

    CUDA_SAFE_CALL(cudaFree(um_clusters));

    // main thread cleanup
    free(fcs_data_by_dimension);

    free(special_memberships);

	*final_num_clusters =ideal_num_clusters;
            return saved_clusters;

    }

    private void memcpy() {}

    private void seed_clusters() {}

    public static void main(String[]args)throws FileNotFoundException{
            // Getting the context info from a file in json
        String CONFIG_PATH="config/config.json";
        Gson gson=new GsonBuilder().setPrettyPrinting().create();
        JsonReader reader=new JsonReader(new FileReader(CONFIG_PATH));
        Config parsedConfig=gson.fromJson(reader,Config.class);

        // Prepare the kernels
        GMM gmm=new GMM(parsedConfig);

        //////////////////////////
        //     Program main     //
        //////////////////////////
        int original_num_clusters,desired_num_clusters;
        String fileName;

        // Command input simulation
        original_num_clusters=16;
        desired_num_clusters=8;
        fileName="../data/mydata.txt";

        float[]fcs_data_by_event=gmm.readData(fileName);

        cluster_t clusters=gmm.cluster();

        // TODO: print the results
    }

    class cluster_t {
        float[] N;
        float[] pi;
        float[] constant;
        float[] avgvar;
        float[] means;
        float[] R;
        float[] Rinv;
        float[] memberships;
    }

    class cluster_um {
        Value N;
        Value pi;
        Value constant;
        Value means;
        Value R;
        Value Rinv;
        Value memberships;
    }

    private Context createContext(Config config) {
        return Context
                .newBuilder()
                .allowAllAccess(true)
                .allowExperimentalOptions(true)
                //logging settings
                .option("log.grcuda.com.nvidia.grcuda.level", "WARNING")
                .option("log.grcuda.com.nvidia.grcuda.GrCUDAContext.level", "SEVERE")
                //GrCUDA env settings
                .option("grcuda.ExecutionPolicy", config.exec_policies)
                .option("grcuda.InputPrefetch", String.valueOf(config.prefetch))
                .option("grcuda.RetrieveNewStreamPolicy", config.new_stream_policies)
                .option("grcuda.RetrieveParentStreamPolicy", config.parent_stream_policies)
                .option("grcuda.DependencyPolicy", config.dependency_policies)
                .option("grcuda.DeviceSelectionPolicy", config.choose_device_policies)
                .option("grcuda.ForceStreamAttach", String.valueOf(config.stream_attach))
                .option("grcuda.EnableComputationTimers", String.valueOf(config.time_computation))
                .option("grcuda.MemAdvisePolicy", config.memory_advise)
                .option("grcuda.NumberOfGPUs", String.valueOf(config.num_gpus))
                .option("grcuda.BandwidthMatrix", config.bandwidthMatrix)
                .build();
    }
}
