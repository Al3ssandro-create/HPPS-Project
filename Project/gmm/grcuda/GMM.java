import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.stream.JsonReader;

import java.awt.image.AreaAveragingScaleFilter;
import java.io.*;
import java.util.ArrayList;
import java.util.Scanner;

public class GMM {
    private Context context;

    private int TRUNCATE = 1;

    private int num_dimensions;
    private int num_events;
    private int ideal_num_clusters;

    private Value seed_clusters;
    private Value constants_kernel;
    private Value estep1;
    private Value estep2;
    private Value mstep_N;
    private Value mstep_means;
    private Value mstep_covariance2;

    public GMM(Config config) {
        this.context = createContext(config);

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

    private float[] readCSV(String fileName)  {
        int num_dims = 0;
        String line1;
        ArrayList<String> lines = new ArrayList<>();

        try(Scanner scanner = new Scanner(new File(fileName))) {
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

    private cluster_t cluster() {
        // TODO
        return null;
    }

    public static void main(String[] args) throws FileNotFoundException {
        // Getting the context info from a file in json
        String CONFIG_PATH = "config/config.json";
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        JsonReader reader = new JsonReader(new FileReader(CONFIG_PATH));
        Config parsedConfig = gson.fromJson(reader, Config.class);

        // Prepare the kernels
        GMM gmm = new GMM(parsedConfig);

        //////////////////////////
        //     Program main     //
        //////////////////////////
        int original_num_clusters, desired_num_clusters;
        String fileName;

        // Command input simulation
        original_num_clusters = 16;
        desired_num_clusters = 8;
        fileName = "../data/mydata.txt";

        float[] fcs_data_by_event = gmm.readData(fileName);

        cluster_t clusters = gmm.cluster();

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
