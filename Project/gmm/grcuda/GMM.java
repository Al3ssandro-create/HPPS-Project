import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.stream.JsonReader;

import java.io.*;

public class GMM {
    private Context contex;
    private int num_dimensions;
    private int num_events;
    private int ideal_num_clusters;

    public GMM(Config config) {
        this.contex = createContext(config);
    }

    private void buildKernels(){
        // TODO
    }

    private float[] readData(String fileName) {
        // TODO
        return null;
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
        gmm.buildKernels();

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