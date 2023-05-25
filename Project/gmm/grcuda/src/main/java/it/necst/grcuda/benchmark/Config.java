package it.necst.grcuda.benchmark;
public class Config {
    public String exec_policies;
    public String dependency_policies;
    public String new_stream_policies;
    public String parent_stream_policies;
    public String choose_device_policies;
    public String memory_advise;
    public boolean prefetch;
    public boolean stream_attach;
    public boolean time_computation;
    public int num_gpus;
    public String bandwidthMatrix = "";
    // public String bandwidthMatrix = "~/grcuda/projects/resources/connection_graph/datasets/connection_graph_test.csv";
}
