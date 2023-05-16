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
    private int PROFILING = 1;
    private int NUM_BLOCKS = 24;
    private int NUM_CLUSTERS_PER_BLOCK = 6;
    private int NUM_THREADS_MSTEP = 256;
    private int NUM_THREADS_ESTEP = 256;
    private int MIN_ITERS = 1;
    private int MAX_ITERS = 200;

    private int num_gpus;
    private int num_dimensions;
    private int num_events;
    private int ideal_num_clusters = 0;
    private int final_num_clusters;

    private Value seed_clusters;
    private Value constants_kernel;
    private Value estep1;
    private Value estep2;
    private Value mstep_N;
    private Value mstep_means;
    private Value mstep_covariance2;

    private long start_exec;
    private long end_exec;

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

    public float[] readData(String fileName) {
        int length = fileName.length();
        System.out.println("File Extension: " + fileName.substring(length - 3));
        if (fileName.endsWith("bin")) {
            return readBIN(fileName);
        } else {
            return readCSV(fileName);
        }
    }

    public float[] readBIN(String fileName) {
        // TODO
        return null;
    }

    public float[] readCSV(String fileName) {
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

    public cluster_t cluster(int original_num_clusters, int desired_num_clusters, float[] fcs_data_by_event) {
        int regroup_iterations = 0;
        int params_iterations = 0;
        int constants_iterations = 0;
        int reduce_iterations = 0;
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
        cluster_um[] um_clusters = new cluster_um[num_gpus];

        // Only need one copy of all the memberships
        float[] special_memberships = new float[num_events * original_num_clusters];

        // Declare another set of clusters for saving the results of the best configuration
        cluster_t saved_clusters = new cluster_t();
        saved_clusters.N = new float[original_num_clusters];
        saved_clusters.pi = new float[original_num_clusters];
        saved_clusters.constant = new float[original_num_clusters];
        saved_clusters.avgvar = new float[original_num_clusters];
        saved_clusters.means = new float[num_dimensions * original_num_clusters];
        saved_clusters.R = new float[num_dimensions * num_dimensions * original_num_clusters];
        saved_clusters.Rinv = new float[num_dimensions * num_dimensions * original_num_clusters];
        saved_clusters.memberships = new float[num_events * original_num_clusters];

        if (DEBUG == 1) System.out.println("Finished allocating shared cluster structures on host");

        // TODO: Probably we can do an array of size num gpus (normal) and then each el is a Value of NUM_BLOCK size
        // Used to hold the result from regroup kernel
        Value[] shared_likelihoods = new Value[NUM_BLOCKS * num_gpus];
        float likelihood, old_likelihood;
        float min_rissanen = 0;

        // ------ OPENMP START ------

        // TODO: check here the size, they are different from 'saved_clusters'
        cluster_t scratch_cluster = new cluster_t();
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

            if (DEBUG == 1) System.out.println("\tN: " + um_clusters[0].N.getArrayElement(c).asString());

            if (DEBUG == 1) System.out.println("\tpi: " + um_clusters[0].pi.getArrayElement(c).asString());

            if (DEBUG == 1) System.out.println("\tMeans:");
            for (int d = 0; d < num_dimensions; d++)
                if (DEBUG == 1) System.out.println("\t\t" + um_clusters[0].means.getArrayElement(c * num_dimensions + d).asString());

            if (DEBUG == 1) System.out.println("\tR:");
            for (int d = 0; d < num_dimensions; d++)
                for (int e = 0; e < num_dimensions; e++)
                    if (DEBUG == 1) System.out.println("\t\t" + um_clusters[0].R.getArrayElement(c * num_dimensions * num_dimensions + d * num_dimensions + e).asString());

            if (DEBUG == 1) System.out.println("\tR-inverse:");
            for (int d = 0; d < num_dimensions; d++)
                for (int e = 0; e < num_dimensions; e++)
                    if (DEBUG == 1) System.out.println("\t\t" + um_clusters[0].Rinv.getArrayElement(c * num_dimensions * num_dimensions + d * num_dimensions + e).asString());

            if (DEBUG == 1) System.out.println("\tAvgvar: " + um_clusters[0].avgvar.getArrayElement(c).asString());

            if (DEBUG == 1) System.out.println("\tConstant: " + um_clusters[0].constant.getArrayElement(c).asString());
        }
        // ------ OPENMP MASTER END (NUM_GPU == 0) ------

        // Synchronize after first gpu does the seeding, copy result to all gpus
        for (int i = 0; i < num_gpus; i++) {
            //memcpy(um_clusters[tid].N, um_clusters[0].N, sizeof( float)*original_num_clusters);
            //memcpy(um_clusters[tid].pi, um_clusters[0].pi, sizeof( float)*original_num_clusters);
            //memcpy(um_clusters[tid].constant, um_clusters[0].constant, sizeof( float)*original_num_clusters);
            //memcpy(um_clusters[tid].avgvar, um_clusters[0].avgvar, sizeof( float)*original_num_clusters);
            //memcpy(um_clusters[tid].means, um_clusters[0].means, sizeof( float)*num_dimensions * original_num_clusters);
            //memcpy(um_clusters[tid].R, um_clusters[0].R, sizeof( float)*num_dimensions * num_dimensions * original_num_clusters);
            //memcpy(um_clusters[tid].Rinv, um_clusters[0].Rinv, sizeof( float)*num_dimensions * num_dimensions * original_num_clusters);
            memcpy();
            memcpy();
            memcpy();
            memcpy();
            memcpy();
            memcpy();
            memcpy();
        }

        // Calculate an epsilon value
        float epsilon = (1 + num_dimensions + 0.5f * (num_dimensions + 1) * num_dimensions) * (float) Math.log((double) num_events * num_dimensions) * 0.001f;
        int iters;

        //epsilon = 1e-6;
        if (PRINT == 1) System.out.println("Gaussian.cu: epsilon = " + epsilon);

        // Variables for GMM reduce order
        float distance, min_distance = 0.0F;
        float rissanen;
        int min_c1, min_c2;

        start_exec = System.nanoTime();

        for (int num_clusters = original_num_clusters; num_clusters >= stop_number; num_clusters--) {
            /*************** EM ALGORITHM *****************************/

            // Do initial E-step
            // Calculates a cluster membership probability
            // for each event and each cluster.
            if (DEBUG == 1) System.out.println("Invoking E-step kernels.");
            int[] gridDim = {num_clusters, NUM_BLOCKS};
            for (int i = 0; i < num_gpus; i++) {
                estep1.execute(gridDim, NUM_THREADS_ESTEP)
                        .execute(um_fcs_data_by_dimension[i], um_clusters[i].means, um_clusters[i].Rinv, um_clusters[i].pi,
                                um_clusters[i].constant, um_clusters[i].memberships, num_dimensions, my_num_events[i]);
            }
            for (int i = 0; i < num_gpus; i++) {
                estep2.execute(NUM_BLOCKS, NUM_THREADS_ESTEP)
                        .execute(um_clusters[i].memberships, num_dimensions, num_clusters, my_num_events[i], shared_likelihoods[i * NUM_BLOCKS]);
            }
            regroup_iterations++;

            likelihood = 0.0F;
            for (int i = 0; i < num_gpus; i++)
                for (int j = 0; j < NUM_BLOCKS; j++)
                    likelihood += shared_likelihoods[i].getArrayElement(j).asFloat();

            if (DEBUG == 1) System.out.println("Likelihood: " + likelihood);

            float change = epsilon * 2;

            if (PRINT == 1) System.out.println("Performing EM algorithm on" + num_clusters + "clusters.");
            iters = 0;
            // This is the iterative loop for the EM algorithm.
            // It re-estimates parameters, re-computes constants, and then regroups the events
            // These steps keep repeating until the change in likelihood is less than some epsilon
            while (iters < MIN_ITERS || (Math.abs(change) > epsilon && iters < MAX_ITERS)) {
                old_likelihood = likelihood;

                if (DEBUG == 1) System.out.println("Invoking reestimate_parameters (M-step) kernel.");

                // This kernel computes a new N, pi isn't updated until compute_constants though
                for (int i = 0; i < num_gpus; i++) {
                    mstep_N.execute(num_clusters, NUM_THREADS_MSTEP)
                            .execute(um_clusters[i].memberships, um_clusters[i].N, um_clusters[i].pi, num_dimensions, num_clusters, my_num_events[i]);
                }

                for (int g = 1; g < num_gpus; g++) {
                    for (int c = 0; c < num_clusters; c++) {
                        um_clusters[0].N.setArrayElement(c, um_clusters[0].N.getArrayElement(c).asFloat() + um_clusters[g].N.getArrayElement(c).asFloat());
                        if (DEBUG == 1) System.out.println("Cluster " + c + ": N = " + um_clusters[0].N.getArrayElement(c));
                    }
                }

                for (int i = 0; i < num_gpus; i++) {
                    //memcpy(um_clusters[tid].N, um_clusters[0].N, sizeof( float)*num_clusters);
                    memcpy();
                }

                int[] gridDim1 = {num_clusters, num_dimensions};
                for (int i = 0; i < num_gpus; i++) {
                    mstep_means.execute(gridDim1, NUM_THREADS_MSTEP)
                            .execute(um_fcs_data_by_dimension[i], um_clusters[i].memberships, um_clusters[i].means, num_dimensions, num_clusters, my_num_events[i]);
                }

                // Reduce means for all clusters, copy back to device
                // ------ OPENMP MASTER START (NUM_GPU == 0) ------
                for (int g = 1; g < num_gpus; g++) {
                    for (int c = 0; c < num_clusters; c++) {
                        for (int d = 0; d < num_dimensions; d++) {
                            um_clusters[0].means.setArrayElement(c * num_dimensions + d,
                                    um_clusters[0].means.getArrayElement(c * num_dimensions + d).asFloat() +
                                            um_clusters[g].means.getArrayElement(c * num_dimensions + d).asFloat());
                        }
                    }
                }
                for (int c = 0; c < num_clusters; c++) {
                    if (DEBUG == 1) System.out.println("Cluster " + c + " Means: " + um_clusters[0].N.getArrayElement(c).asString());
                    for (int d = 0; d < num_dimensions; d++) {
                        if (um_clusters[0].N.getArrayElement(c).asFloat() > 0.5F) {
                            um_clusters[0].means.setArrayElement(c * num_dimensions + d,
                                    um_clusters[0].means.getArrayElement(c * num_dimensions + d).asFloat() /
                                            um_clusters[0].N.getArrayElement(c).asFloat());
                        } else {
                            um_clusters[0].means.setArrayElement(c * num_dimensions + d, 0.0F);
                        }
                        if (DEBUG == 1) System.out.println("\t" + um_clusters[0].means.getArrayElement(c * num_dimensions + d).asString());
                    }
                }
                // ------ OPENMP MASTER END (NUM_GPU == 0) ------

                for (int i = 0; i < num_gpus; i++) {
                    // memcpy(um_clusters[tid].means, um_clusters[0].means, sizeof( float)*num_clusters * num_dimensions);
                    memcpy();
                }

                // Covariance is symmetric, so we only need to compute N*(N+1)/2 matrix elements per cluster
                int[] gridDim2 = {(num_clusters + NUM_CLUSTERS_PER_BLOCK - 1) / NUM_CLUSTERS_PER_BLOCK, num_dimensions * (num_dimensions + 1) / 2};
                for (int i = 0; i < num_gpus; i++) {
                    //mstep_covariance1<<<gridDim2, NUM_THREADS_MSTEP>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,my_num_events);
                    mstep_covariance2.execute(gridDim2, NUM_THREADS_MSTEP)
                            .execute(um_fcs_data_by_dimension, um_clusters[i].R, um_clusters[i].means,
                                    um_clusters[i].memberships, um_clusters[i].avgvar, num_dimensions, num_clusters, my_num_events[i]);
                }

                // Reduce R for all clusters, copy back to device
                // ------ OPENMP MASTER START (NUM_GPU == 0) ------
                for (int g = 1; g < num_gpus; g++) {
                    for (int c = 0; c < num_clusters; c++) {
                        for (int d = 0; d < num_dimensions * num_dimensions; d++) {
                            um_clusters[0].R.setArrayElement(c * num_dimensions * num_dimensions + d,
                                    um_clusters[0].R.getArrayElement(c * num_dimensions * num_dimensions + d).asFloat() +
                                            um_clusters[g].R.getArrayElement(c * num_dimensions * num_dimensions + d).asFloat());
                        }
                    }
                }
                for (int c = 0; c < num_clusters; c++) {
                    if (um_clusters[0].N.getArrayElement(c).asFloat() > 0.5F) {
                        for (int d = 0; d < num_dimensions * num_dimensions; d++) {
                            um_clusters[0].R.setArrayElement(c * num_dimensions * num_dimensions + d,
                                    um_clusters[0].R.getArrayElement(c * num_dimensions * num_dimensions + d).asFloat() /
                                            um_clusters[0].N.getArrayElement(c).asFloat());
                        }
                    } else {
                        for (int i = 0; i < num_dimensions; i++) {
                            for (int j = 0; j < num_dimensions; j++) {
                                if (i == j) {
                                    um_clusters[0].R.setArrayElement(c * num_dimensions * num_dimensions + i * num_dimensions + j, 1.0F);
                                } else {
                                    um_clusters[0].R.setArrayElement(c * num_dimensions * num_dimensions + i * num_dimensions + j, 0.0F);
                                }
                            }
                        }
                    }
                }
                // ------ OPENMP MASTER END (NUM_GPU == 0) ------

                for (int i = 0; i < num_gpus; i++) {
                    //memcpy(um_clusters[tid].R, um_clusters[0].R, sizeof(float)*num_clusters * num_dimensions * num_dimensions);
                    memcpy();
                }

                params_iterations++;

                if (DEBUG == 1) System.out.println("Invoking constants kernel.");
                // Inverts the R matrices, computes the constant, normalizes cluster probabilities
                for (int i = 0; i < num_gpus; i++) {
                    constants_kernel.execute(num_clusters, NUM_THREADS_MSTEP)
                            .execute(um_clusters[i].R, um_clusters[i].Rinv, um_clusters[i].constant, um_clusters[i].N, num_clusters, num_dimensions);
                }


                for (int temp_c = 0; temp_c < num_clusters; temp_c++)
                    if (DEBUG == 1) System.out.println("Cluster " + temp_c +" constant: " + um_clusters[0].constant.getArrayElement(temp_c).asString());

                constants_iterations++;

                if (DEBUG == 1) System.out.println("Invoking regroup (E-step) kernel with " + NUM_BLOCKS + " blocks.");

                // Compute new cluster membership probabilities for all the events
                int[] gridDim3 = {num_clusters, NUM_BLOCKS};
                for (int i = 0; i < num_gpus; i++) {
                    estep1.execute(gridDim3, NUM_THREADS_ESTEP)
                            .execute(um_fcs_data_by_dimension, um_clusters[i].means, um_clusters[i].Rinv, um_clusters[i].pi, um_clusters[i].constant, num_dimensions, my_num_events[i]);
                }
                for (int i = 0; i < num_gpus; i++) {
                    estep2.execute(NUM_BLOCKS, NUM_THREADS_ESTEP)
                            .execute(um_clusters[i].memberships, num_dimensions, num_clusters, my_num_events[i], shared_likelihoods[i * NUM_BLOCKS]);
                }

                regroup_iterations++;

                // ------ OPENMP MASTER START (NUM_GPU == 0) ------
                likelihood = 0.0F;
                for (int i = 0; i < NUM_BLOCKS * num_gpus; i++)
                    for (int j = 0; j < NUM_BLOCKS; j++)
                        likelihood += shared_likelihoods[i].getArrayElement(j).asFloat();

                if (DEBUG == 1) System.out.println("Likelihood: " + likelihood);
                // ------ OPENMP MASTER END (NUM_GPU == 0) ------

                // TODO: Should be change an array per each gpu??
                change = likelihood - old_likelihood;
                if (DEBUG == 1) System.out.println("GPU " + 0 + ", Change in likelihood: " + change);

                iters++;
            }

            if (DEBUG == 1) System.out.println("Done with EM loop");

            for (int i = 0; i < num_gpus; i++) {
                for (int c = 0; c < num_clusters; c++) {
                    // memcpy( & (special_memberships[c * num_events + tid * events_per_gpu]), &(um_clusters[tid].memberships[c * my_num_events]), sizeof( float)*my_num_events);
                    memcpy();
                }
                if (DEBUG == 1) System.out.println("GPU " + i + "done with copying cluster data from device");
            }

            // Calculate Rissanen Score
            rissanen = -likelihood + 0.5F * (num_clusters * (1.0F + num_dimensions + 0.5F * (num_dimensions + 1.0F) * num_dimensions) - 1.0F) * (float) Math.log((double) num_events * num_dimensions);
            if (PRINT == 1) System.out.println("Likelihood: " + likelihood);
            if (PRINT == 1) System.out.println("Rissanen Score: " + rissanen);

            // ------ OPENMP MASTER START (NUM_GPU == 0) ------
            // Save the cluster data the first time through, so we have a base rissanen score and result
            // Save the cluster data if the solution is better and the user didn't specify a desired number
            // If the num_clusters equals the desired number, stop
            if (num_clusters == original_num_clusters || (rissanen < min_rissanen && desired_num_clusters == 0) || (num_clusters == desired_num_clusters)) {
                min_rissanen = rissanen;
                ideal_num_clusters = num_clusters;
                // Save the cluster configuration somewhere
                // memcpy(saved_clusters -> N, um_clusters[0].N, sizeof( float)*num_clusters);
                // memcpy(saved_clusters -> pi, um_clusters[0].pi, sizeof( float)*num_clusters);
                // memcpy(saved_clusters -> constant, um_clusters[0].constant, sizeof( float)*num_clusters);
                // memcpy(saved_clusters -> avgvar, um_clusters[0].avgvar, sizeof( float)*num_clusters);
                // memcpy(saved_clusters -> means, um_clusters[0].means, sizeof( float)*num_dimensions * num_clusters);
                // memcpy(saved_clusters -> R, um_clusters[0].R, sizeof( float)*num_dimensions * num_dimensions * num_clusters);
                // memcpy(saved_clusters -> Rinv, um_clusters[0].Rinv, sizeof( float)*num_dimensions * num_dimensions * num_clusters);
                // memcpy(saved_clusters -> memberships, special_memberships, sizeof( float)*num_events * num_clusters);
                memcpy();
                memcpy();
                memcpy();
                memcpy();
                memcpy();
                memcpy();
                memcpy();
                memcpy();
            }
            // ------ OPENMP MASTER END (NUM_GPU == 0) ------

            // **************** Reduce GMM Order ********************

            // Don't want to reduce order on the last iteration
            if (num_clusters > stop_number) {
                // ------ OPENMP MASTER START (NUM_GPU == 0) ------
                // First eliminate any "empty" clusters
                for (int i = num_clusters - 1; i >= 0; i--) {
                    if (um_clusters[0].N.getArrayElement(i).asFloat() < 0.5) {
                        if (DEBUG == 1) System.out.println("Cluster #" + i + " has less than 1 data point in it.");
                        for (int j = i; j < num_clusters - 1; j++) {
                            //copy_cluster(um_clusters[0], j, um_clusters[0], j + 1, num_dimensions);
                            copy_cluster();
                        }
                        num_clusters--;
                    }
                }

                min_c1 = 0;
                min_c2 = 1;
                if (DEBUG == 1) System.out.println("Number of non-empty clusters: " + num_clusters);
                // For all combinations of subclasses...
                // If the number of clusters got really big might need to do a non-exhaustive search
                // Even with 100*99/2 combinations this doesn't seem to take too long
                for (int c1 = 0; c1 < num_clusters; c1++) {
                    for (int c2 = c1 + 1; c2 < num_clusters; c2++) {
                        // compute distance function between the 2 clusters
                        //distance = cluster_distance(um_clusters[0], c1, c2, scratch_cluster, num_dimensions);
                        distance = cluster_distance();
                        // Keep track of minimum distance
                        if ((c1 == 0 && c2 == 1) || distance < min_distance) {
                            min_distance = distance;
                            min_c1 = c1;
                            min_c2 = c2;
                        }
                    }
                }

                if (PRINT == 1) System.out.println("Minimum distance between (" + min_c1 + ", " + min_c2 + "). Combining clusters");
                // Add the two clusters with min distance together
                // add_clusters(&(clusters[min_c1]),&(clusters[min_c2]),scratch_cluster,num_dimensions);

                //add_clusters(um_clusters[0], min_c1, min_c2, scratch_cluster, num_dimensions);
                add_clusters();

                // Copy new combined cluster into the main group of clusters, compact them
                //copy_cluster(um_clusters[0], min_c1, scratch_cluster, 0, num_dimensions);
                copy_cluster();

                for (int i = min_c2; i < num_clusters - 1; i++) {
                    // System.out.printf("Copying cluster " + (i+1) + " to cluster " + i);
                    //copy_cluster(um_clusters[0], i, um_clusters[0], i + 1, num_dimensions);
                    copy_cluster();
                }
                // ------ OPENMP MASTER END (NUM_GPU == 0) ------

                // Copy the clusters back to the device
                for (int i = 0; i < num_gpus; i++) {
                    //memcpy(um_clusters[tid].N, um_clusters[0].N, sizeof( float)*num_clusters);
                    //memcpy(um_clusters[tid].pi, um_clusters[0].pi, sizeof( float)*num_clusters);
                    //memcpy(um_clusters[tid].constant, um_clusters[0].constant, sizeof( float)*num_clusters);
                    //memcpy(um_clusters[tid].avgvar, um_clusters[0].avgvar, sizeof( float)*num_clusters);
                    //memcpy(um_clusters[tid].means, um_clusters[0].means, sizeof( float)*num_dimensions * num_clusters);
                    //memcpy(um_clusters[tid].R, um_clusters[0].R, sizeof( float)*num_dimensions * num_dimensions * num_clusters);
                    //memcpy(um_clusters[tid].Rinv, um_clusters[0].Rinv, sizeof( float)*num_dimensions * num_dimensions * num_clusters);
                    memcpy();
                    memcpy();
                    memcpy();
                    memcpy();
                    memcpy();
                    memcpy();
                    memcpy();
                }

                for (int i = 0; i < num_gpus; i++) {
                    for (int c = 0; c < num_clusters; c++) {
                        //memcpy( & um_clusters[tid].memberships[c * my_num_events], &(special_memberships[c * num_events + tid * (num_events / num_gpus)]), sizeof( float)*my_num_events);
                        memcpy();
                    }
                }
            } // GMM reduction block

            reduce_iterations++;
        } // outer loop from M to 1 clusters

        end_exec = System.nanoTime();

        if (PRINT == 1) System.out.println("Final rissanen Score was: " + min_rissanen + ", with " + ideal_num_clusters + "clusters.");
        if (PROFILING == 1) System.out.println("##### Time execution overall : " + (end_exec - start_exec) * 1000 +" micro sec. #####");
        // ------ OPENMP START ------

	    final_num_clusters = ideal_num_clusters;
        return saved_clusters;
    }

    public void memcpy() {}

    public void seed_clusters() {}

    public void copy_cluster() {}

    public float cluster_distance() {
        return 0;
    }

    public void add_clusters() {}

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

        cluster_t clusters = gmm.cluster(original_num_clusters, desired_num_clusters, fcs_data_by_event);

        // TODO: print the results
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

class cluster_t{
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
    Value avgvar;
    Value means;
    Value R;
    Value Rinv;
    Value memberships;
}
