package it.necst.grcuda.benchmark;
public class GMMKernels {
    static String constants_kernel = "\n" +
            "#define PI  3.1415926535897931\n" +
            "#define NUM_DIMENSIONS 24\n" +
            "#define DIAG_ONLY 0\n" +
            "\n" +
            "__device__ void compute_pi(float* c_N, float* c_pi, int num_clusters) {\n" +
            "    __shared__ float sum;\n" +
            "    \n" +
            "    if(threadIdx.x == 0) {\n" +
            "        sum = 0.0;\n" +
            "        for(int i=0; i<num_clusters; i++) {\n" +
            "            sum += c_N[i];\n" +
            "        }\n" +
            "    }\n" +
            "    \n" +
            "    __syncthreads();\n" +
            "    \n" +
            "    for(int c=threadIdx.x; c < num_clusters; c += blockDim.x) {\n" +
            "        if(c_N[c] < 0.5f) {\n" +
            "            c_pi[threadIdx.x] = 1e-10;\n" +
            "        } else {\n" +
            "            c_pi[threadIdx.x] = c_N[c] / sum;\n" +
            "        }\n" +
            "    }\n" +
            " \n" +
            "    __syncthreads();\n" +
            "}\n" +
            "\n" +
            "__device__ void invert(float* data, int actualsize, float* log_determinant)  {\n" +
            "    int maxsize = actualsize;\n" +
            "    int n = actualsize;\n" +
            "    \n" +
            "    if(threadIdx.x == 0) {\n" +
            "        *log_determinant = 0.0;\n" +
            "\n" +
            "      // sanity check        \n" +
            "      if (actualsize == 1) {\n" +
            "        *log_determinant = logf(data[0]);\n" +
            "        data[0] = 1.0 / data[0];\n" +
            "      } else {\n" +
            "\n" +
            "          for (int i=1; i < actualsize; i++) data[i] /= data[0]; // normalize row 0\n" +
            "          for (int i=1; i < actualsize; i++)  { \n" +
            "            for (int j=i; j < actualsize; j++)  { // do a column of L\n" +
            "              float sum = 0.0;\n" +
            "              for (int k = 0; k < i; k++)  \n" +
            "                  sum += data[j*maxsize+k] * data[k*maxsize+i];\n" +
            "              data[j*maxsize+i] -= sum;\n" +
            "              }\n" +
            "            if (i == actualsize-1) continue;\n" +
            "            for (int j=i+1; j < actualsize; j++)  {  // do a row of U\n" +
            "              float sum = 0.0;\n" +
            "              for (int k = 0; k < i; k++)\n" +
            "                  sum += data[i*maxsize+k]*data[k*maxsize+j];\n" +
            "              data[i*maxsize+j] = \n" +
            "                 (data[i*maxsize+j]-sum) / data[i*maxsize+i];\n" +
            "              }\n" +
            "            }\n" +
            "            \n" +
            "            for(int i=0; i<actualsize; i++) {\n" +
            "                *log_determinant += logf(fabs(data[i*n+i]));\n" +
            "            }\n" +
            "            \n" +
            "          for ( int i = 0; i < actualsize; i++ )  // invert L\n" +
            "            for ( int j = i; j < actualsize; j++ )  {\n" +
            "              float x = 1.0;\n" +
            "              if ( i != j ) {\n" +
            "                x = 0.0;\n" +
            "                for ( int k = i; k < j; k++ ) \n" +
            "                    x -= data[j*maxsize+k]*data[k*maxsize+i];\n" +
            "                }\n" +
            "              data[j*maxsize+i] = x / data[j*maxsize+j];\n" +
            "              }\n" +
            "          for ( int i = 0; i < actualsize; i++ )   // invert U\n" +
            "            for ( int j = i; j < actualsize; j++ )  {\n" +
            "              if ( i == j ) continue;\n" +
            "              float sum = 0.0;\n" +
            "              for ( int k = i; k < j; k++ )\n" +
            "                  sum += data[k*maxsize+j]*( (i==k) ? 1.0 : data[i*maxsize+k] );\n" +
            "              data[i*maxsize+j] = -sum;\n" +
            "              }\n" +
            "          for ( int i = 0; i < actualsize; i++ )   // final inversion\n" +
            "            for ( int j = 0; j < actualsize; j++ )  {\n" +
            "              float sum = 0.0;\n" +
            "              for ( int k = ((i>j)?i:j); k < actualsize; k++ )  \n" +
            "                  sum += ((j==k)?1.0:data[j*maxsize+k])*data[k*maxsize+i];\n" +
            "              data[j*maxsize+i] = sum;\n" +
            "              }\n" +
            "        }\n" +
            "    }\n" +
            " }\n" +
            "\n" +
            "__device__ void compute_constants(float* c_R, float* c_Rinv, float* c_constant, int num_clusters, int num_dimensions) {\n" +
            "    int tid = threadIdx.x;\n" +
            "    int num_threads = blockDim.x;\n" +
            "    int num_elements = num_dimensions*num_dimensions;\n" +
            "    \n" +
            "    __shared__ float determinant_arg; // only one thread computes the inverse so we need a shared argument\n" +
            "    \n" +
            "    float log_determinant;\n" +
            "    \n" +
            "    __shared__ float matrix[NUM_DIMENSIONS*NUM_DIMENSIONS];\n" +
            "    \n" +
            "    // Invert the matrix for every cluster\n" +
            "    int c = blockIdx.x;\n" +
            "    // Copy the R matrix into shared memory for doing the matrix inversion\n" +
            "    for(int i=tid; i<num_elements; i+= num_threads ) {\n" +
            "        matrix[i] = c_R[c*num_dimensions*num_dimensions+i];\n" +
            "    }\n" +
            "    \n" +
            "    __syncthreads(); \n" +
            "    #if DIAG_ONLY\n" +
            "        if(tid == 0) { \n" +
            "            determinant_arg = 1.0f;\n" +
            "            for(int i=0; i < num_dimensions; i++) {\n" +
            "                determinant_arg *= matrix[i*num_dimensions+i];\n" +
            "                matrix[i*num_dimensions+i] = 1.0f / matrix[i*num_dimensions+i];\n" +
            "            }\n" +
            "            determinant_arg = logf(determinant_arg);\n" +
            "        }\n" +
            "    #else \n" +
            "        invert(matrix,num_dimensions,&determinant_arg);\n" +
            "    #endif\n" +
            "    __syncthreads(); \n" +
            "    log_determinant = determinant_arg;\n" +
            "    \n" +
            "    // Copy the matrx from shared memory back into the cluster memory\n" +
            "    for(int i=tid; i<num_elements; i+= num_threads) {\n" +
            "        c_Rinv[c*num_dimensions*num_dimensions+i] = matrix[i];\n" +
            "    }\n" +
            "    \n" +
            "    __syncthreads();\n" +
            "    \n" +
            "    // Compute the constant\n" +
            "    // Equivilent to: log(1/((2*PI)^(M/2)*det(R)^(1/2)))\n" +
            "    // This constant is used in all E-step likelihood calculations\n" +
            "    if(tid == 0) {\n" +
            "        c_constant[c] = -num_dimensions*0.5f*logf(2.0f*PI) - 0.5f*log_determinant;\n" +
            "    }\n" +
            "}\n" +
            "\n" +
            "__global__ void constants_kernel(float* c_R, float* c_Rinv, float* c_constant, float* c_N, float* c_pi, int num_clusters, int num_dimensions) {\n" +
            "    compute_constants(c_R,c_Rinv,c_constant,num_clusters,num_dimensions);\n" +
            "    \n" +
            "    __syncthreads();\n" +
            "    \n" +
            "    if(blockIdx.x == 0) {\n" +
            "        compute_pi(c_N,c_pi,num_clusters);\n" +
            "    }\n" +
            "}";

    static String seed_clusters = "\n" +
            "#define COVARIANCE_DYNAMIC_RANGE 1E6\n" +
            "#define NUM_DIMENSIONS 24\n" +
            "\n" +
            "__device__ void mvtmeans(float* fcs_data, int num_dimensions, int num_events, float* means) {\n" +
            "    // access thread id\n" +
            "    int tid = threadIdx.x;\n" +
            "\n" +
            "    if(tid < num_dimensions) {\n" +
            "        means[tid] = 0.0;\n" +
            "\n" +
            "        // Sum up all the values for the dimension\n" +
            "        for(int i=0; i < num_events; i++) {\n" +
            "            means[tid] += fcs_data[i*num_dimensions+tid];\n" +
            "        }\n" +
            "\n" +
            "        // Divide by the # of elements to get the average\n" +
            "        means[tid] /= (float) num_events;\n" +
            "    }\n" +
            "}\n" +
            "\n" +
            "__device__ void averageVariance(float* fcs_data, float* means, int num_dimensions, int num_events, float* avgvar) {\n" +
            "    // access thread id\n" +
            "    int tid = threadIdx.x;\n" +
            "    \n" +
            "    __shared__ float variances[NUM_DIMENSIONS];\n" +
            "    __shared__ float total_variance;\n" +
            "    \n" +
            "    // Compute average variance for each dimension\n" +
            "    if(tid < num_dimensions) {\n" +
            "        variances[tid] = 0.0;\n" +
            "        // Sum up all the variance\n" +
            "        for(int j=0; j < num_events; j++) {\n" +
            "            // variance = (data - mean)^2\n" +
            "            variances[tid] += (fcs_data[j*num_dimensions + tid])*(fcs_data[j*num_dimensions + tid]);\n" +
            "        }\n" +
            "        variances[tid] /= (float) num_events;\n" +
            "        variances[tid] -= means[tid]*means[tid];\n" +
            "    }\n" +
            "    \n" +
            "    __syncthreads();\n" +
            "    \n" +
            "    if(tid == 0) {\n" +
            "        total_variance = 0.0;\n" +
            "        for(int i=0; i<num_dimensions;i++) {\n" +
            "            ////printf(\"%f \",variances[tid]);\n" +
            "            total_variance += variances[i];\n" +
            "        }\n" +
            "        ////printf(\"\\nTotal variance: %f\\n\",total_variance);\n" +
            "        *avgvar = total_variance / (float) num_dimensions;\n" +
            "        ////printf(\"Average Variance: %f\\n\",*avgvar);\n" +
            "    }\n" +
            "}\n" +
            "__global__ void seed_clusters(float* fcs_data, float* c_means, float* c_R, float* c_N, float* c_pi, float* c_avgvar, int num_dimensions, int num_clusters, int num_events) \n" +
            "{\n" +
            "    // access thread id\n" +
            "    int tid = threadIdx.x;\n" +
            "    // access number of threads in this block\n" +
            "    int num_threads = blockDim.x;\n" +
            "\n" +
            "    // shared memory\n" +
            "    __shared__ float means[NUM_DIMENSIONS];\n" +
            "    \n" +
            "    // Compute the means\n" +
            "    mvtmeans(fcs_data, num_dimensions, num_events, means);\n" +
            "\n" +
            "    __syncthreads();\n" +
            "    \n" +
            "    __shared__ float avgvar;\n" +
            "    \n" +
            "    // Compute the average variance\n" +
            "    averageVariance(fcs_data, means, num_dimensions, num_events, &avgvar);\n" +
            "        \n" +
            "    int num_elements;\n" +
            "    int row, col;\n" +
            "        \n" +
            "    // Number of elements in the covariance matrix\n" +
            "    num_elements = num_dimensions*num_dimensions; \n" +
            "\n" +
            "    __syncthreads();\n" +
            "\n" +
            "    float seed;\n" +
            "    if(num_clusters > 1) {\n" +
            "        seed = (num_events-1.0f)/(num_clusters-1.0f);\n" +
            "    } else {\n" +
            "        seed = 0.0;\n" +
            "    }\n" +
            "    \n" +
            "    // Seed the pi, means, and covariances for every cluster\n" +
            "    for(int c=0; c < num_clusters; c++) {\n" +
            "        if(tid < num_dimensions) {\n" +
            "            c_means[c*num_dimensions+tid] = fcs_data[((int)(c*seed))*num_dimensions+tid];\n" +
            "        }\n" +
            "          \n" +
            "        for(int i=tid; i < num_elements; i+= num_threads) {\n" +
            "            // Add the average variance divided by a constant, this keeps the cov matrix from becoming singular\n" +
            "            row = (i) / num_dimensions;\n" +
            "            col = (i) % num_dimensions;\n" +
            "\n" +
            "            if(row == col) {\n" +
            "                c_R[c*num_dimensions*num_dimensions+i] = 1.0f;\n" +
            "            } else {\n" +
            "                c_R[c*num_dimensions*num_dimensions+i] = 0.0f;\n" +
            "            }\n" +
            "        }\n" +
            "        if(tid == 0) {\n" +
            "            c_pi[c] = 1.0f/((float)num_clusters);\n" +
            "            c_N[c] = ((float) num_events) / ((float)num_clusters);\n" +
            "            c_avgvar[c] = avgvar / COVARIANCE_DYNAMIC_RANGE;\n" +
            "        }\n" +
            "    }\n" +
            "}";

    static String estep1 = "\n" +
            "#define\tNUM_BLOCKS 24\n" +
            "#define NUM_THREADS_ESTEP 256 // should be a power of 2 for parallel reductions to work\n" +
            "#define NUM_DIMENSIONS 24\n" +
            "#define DIAG_ONLY 0\n" +
            "\n" +
            "__device__ void compute_indices(int num_events, int* start, int* stop) {\n" +
            "    // Break up the events evenly between the blocks\n" +
            "    int num_pixels_per_block = num_events / NUM_BLOCKS;\n" +
            "    // Make sure the events being accessed by the block are aligned to a multiple of 16\n" +
            "    num_pixels_per_block = num_pixels_per_block - (num_pixels_per_block % 16);\n" +
            "    \n" +
            "    *start = blockIdx.y * num_pixels_per_block + threadIdx.x;\n" +
            "    \n" +
            "    // Last block will handle the leftover events\n" +
            "    if(blockIdx.y == gridDim.y-1) {\n" +
            "        *stop = num_events;\n" +
            "    } else { \n" +
            "        *stop = (blockIdx.y+1) * num_pixels_per_block;\n" +
            "    }\n" +
            "}\n" +
            "\n" +
            "__global__ void estep1(float* data, float* c_means, float* c_Rinv, float* c_pi, float* c_constant, float* c_memberships, int num_dimensions, int num_events) {\n" +
            "    \n" +
            "    // Cached cluster parameters\n" +
            "    __shared__ float means[NUM_DIMENSIONS];\n" +
            "    __shared__ float Rinv[NUM_DIMENSIONS*NUM_DIMENSIONS];\n" +
            "    float cluster_pi;\n" +
            "    float constant;\n" +
            "    const unsigned int tid = threadIdx.x;\n" +
            " \n" +
            "    int start_index;\n" +
            "    int end_index;\n" +
            "\n" +
            "    int c = blockIdx.x;\n" +
            "\n" +
            "    compute_indices(num_events,&start_index,&end_index);\n" +
            "    \n" +
            "    float like;\n" +
            "\n" +
            "    // This loop computes the expectation of every event into every cluster\n" +
            "    //\n" +
            "    // P(k|n) = L(x_n|mu_k,R_k)*P(k) / P(x_n)\n" +
            "    //\n" +
            "    // Compute log-likelihood for every cluster for each event\n" +
            "    // L = constant*exp(-0.5*(x-mu)*Rinv*(x-mu))\n" +
            "    // log_L = log_constant - 0.5*(x-u)*Rinv*(x-mu)\n" +
            "    // the constant stored in clusters[c].constant is already the log of the constant\n" +
            "    \n" +
            "    // copy the means for this cluster into shared memory\n" +
            "    if(tid < num_dimensions) {\n" +
            "        means[tid] = c_means[c*num_dimensions+tid];\n" +
            "    }\n" +
            "\n" +
            "    // copy the covariance inverse into shared memory\n" +
            "    for(int i=tid; i < num_dimensions*num_dimensions; i+= NUM_THREADS_ESTEP) {\n" +
            "        Rinv[i] = c_Rinv[c*num_dimensions*num_dimensions+i]; \n" +
            "    }\n" +
            "    \n" +
            "    cluster_pi = c_pi[c];\n" +
            "    constant = c_constant[c];\n" +
            "\n" +
            "    // Sync to wait for all params to be loaded to shared memory\n" +
            "    __syncthreads();\n" +
            "    \n" +
            "    for(int event=start_index; event<end_index; event += NUM_THREADS_ESTEP) {\n" +
            "       like = 0.0f;\n" +
            "        // this does the loglikelihood calculation\n" +
            "        #if DIAG_ONLY\n" +
            "            for(int j=0; j<num_dimensions; j++) {\n" +
            "                like += (data[j*num_events+event]-means[j]) * (data[j*num_events+event]-means[j]) * Rinv[j*num_dimensions+j];\n" +
            "            }\n" +
            "        #else\n" +
            "            for(int i=0; i<num_dimensions; i++) {\n" +
            "                for(int j=0; j<num_dimensions; j++) {\n" +
            "                    like += (data[i*num_events+event]-means[i]) * (data[j*num_events+event]-means[j]) * Rinv[i*num_dimensions+j];\n" +
            "                }\n" +
            "            }\n" +
            "        #endif\n" +
            "        // numerator of the E-step probability computation\n" +
            "        c_memberships[c*num_events+event] = -0.5f * like + constant + logf(cluster_pi);\n" +
            "    }\n" +
            "}";

    static String estep2 = "\n" +
            "#define NUM_THREADS_ESTEP 256 // should be a power of 2 for parallel reductions to work\n" +
            "\n" +
            "__device__ float parallelSum(float* data, const unsigned int ndata) {\n" +
            "  const unsigned int tid = threadIdx.x;\n" +
            "  float t;\n" +
            "\n" +
            "  __syncthreads();\n" +
            "\n" +
            "  // Butterfly sum.  ndata MUST be a power of 2.\n" +
            "  for(unsigned int bit = ndata >> 1; bit > 0; bit >>= 1) {\n" +
            "    t = data[tid] + data[tid^bit];  __syncthreads();\n" +
            "    data[tid] = t;                  __syncthreads();\n" +
            "  }\n" +
            "  return data[tid];\n" +
            "}\n" +
            "\n" +
            "__global__ void estep2(float* c_memberships, int num_dimensions, int num_clusters, int num_events, float* likelihood) {\n" +
            "    float temp;\n" +
            "    float thread_likelihood = 0.0f;\n" +
            "    __shared__ float total_likelihoods[NUM_THREADS_ESTEP];\n" +
            "    float max_likelihood;\n" +
            "    float denominator_sum;\n" +
            "    \n" +
            "    // Break up the events evenly between the blocks\n" +
            "    int num_pixels_per_block = num_events / gridDim.x;\n" +
            "    // Make sure the events being accessed by the block are aligned to a multiple of 16\n" +
            "    num_pixels_per_block = num_pixels_per_block - (num_pixels_per_block % 16);\n" +
            "    int tid = threadIdx.x;\n" +
            "    \n" +
            "    int start_index;\n" +
            "    int end_index;\n" +
            "    start_index = blockIdx.x * num_pixels_per_block + tid;\n" +
            "    \n" +
            "    // Last block will handle the leftover events\n" +
            "    if(blockIdx.x == gridDim.x-1) {\n" +
            "        end_index = num_events;\n" +
            "    } else {\n" +
            "        end_index = (blockIdx.x+1) * num_pixels_per_block;\n" +
            "    }\n" +
            "    \n" +
            "    total_likelihoods[tid] = 0.0;\n" +
            "\n" +
            "    // P(x_n) = sum of likelihoods weighted by P(k) (their probability, cluster[c].pi)\n" +
            "    //  log(a+b) != log(a) + log(b) so we need to do the log of the sum of the exponentials\n" +
            "\n" +
            "    //  For the sake of numerical stability, we first find the max and scale the values\n" +
            "    //  That way, the maximum value ever going into the exp function is 0 and we avoid overflow\n" +
            "\n" +
            "    //  log-sum-exp formula:\n" +
            "    //  log(sum(exp(x_i)) = max(z) + log(sum(exp(z_i-max(z))))\n" +
            "    for(int pixel=start_index; pixel<end_index; pixel += NUM_THREADS_ESTEP) {\n" +
            "        // find the maximum likelihood for this event\n" +
            "        max_likelihood = c_memberships[pixel];\n" +
            "        for(int c=1; c<num_clusters; c++) {\n" +
            "            max_likelihood = fmaxf(max_likelihood,c_memberships[c*num_events+pixel]);\n" +
            "        }\n" +
            "\n" +
            "        // Compute P(x_n), the denominator of the probability (sum of weighted likelihoods)\n" +
            "        denominator_sum = 0.0;\n" +
            "        for(int c=0; c<num_clusters; c++) {\n" +
            "            temp = expf(c_memberships[c*num_events+pixel]-max_likelihood);\n" +
            "            denominator_sum += temp;\n" +
            "        }\n" +
            "        denominator_sum = max_likelihood + logf(denominator_sum);\n" +
            "        thread_likelihood += denominator_sum;\n" +
            "        \n" +
            "        // Divide by denominator, also effectively normalize probabilities\n" +
            "        // exp(log(p) - log(denom)) == p / denom\n" +
            "        for(int c=0; c<num_clusters; c++) {\n" +
            "            c_memberships[c*num_events+pixel] = expf(c_memberships[c*num_events+pixel] - denominator_sum);\n" +
            "            //printf(\"Probability that pixel #%d is in cluster #%d: %f\\n\",pixel,c,c_memberships[c*num_events+pixel]);\n" +
            "        }\n" +
            "    }\n" +
            "    \n" +
            "    total_likelihoods[tid] = thread_likelihood;\n" +
            "    __syncthreads();\n" +
            "\n" +
            "    temp = parallelSum(total_likelihoods,NUM_THREADS_ESTEP);\n" +
            "    if(tid == 0) {\n" +
            "        likelihood[blockIdx.x] = temp;\n" +
            "    }\n" +
            "}";

    static String mstep_means = "\n" +
            "#define NUM_THREADS_MSTEP 256 // should be a power of 2 for parallel reductions to work\n" +
            "\n" +
            "__device__ float parallelSum(float* data, const unsigned int ndata) {\n" +
            "  const unsigned int tid = threadIdx.x;\n" +
            "  float t;\n" +
            "\n" +
            "  __syncthreads();\n" +
            "\n" +
            "  // Butterfly sum.  ndata MUST be a power of 2.\n" +
            "  for(unsigned int bit = ndata >> 1; bit > 0; bit >>= 1) {\n" +
            "    t = data[tid] + data[tid^bit];  __syncthreads();\n" +
            "    data[tid] = t;                  __syncthreads();\n" +
            "  }\n" +
            "  return data[tid];\n" +
            "}\n" +
            "\n" +
            "__global__ void mstep_means(float* fcs_data, float* c_memberships, float* c_means, int num_dimensions, int num_clusters, int num_events) {\n" +
            "    // One block per cluster, per dimension:  (M x D) grid of blocks\n" +
            "    int tid = threadIdx.x;\n" +
            "    int num_threads = blockDim.x;\n" +
            "    int c = blockIdx.x; // cluster number\n" +
            "    int d = blockIdx.y; // dimension number\n" +
            "\n" +
            "    __shared__ float temp_sum[NUM_THREADS_MSTEP];\n" +
            "    float sum = 0.0f;\n" +
            "    \n" +
            "    for(int event=tid; event < num_events; event+= num_threads) {\n" +
            "        sum += fcs_data[d*num_events+event]*c_memberships[c*num_events+event];\n" +
            "    }\n" +
            "    temp_sum[tid] = sum;\n" +
            "    \n" +
            "    __syncthreads();\n" +
            "\n" +
            "    // Reduce partial sums\n" +
            "    sum = parallelSum(temp_sum,NUM_THREADS_MSTEP);\n" +
            "    if(tid == 0) {\n" +
            "        c_means[c*num_dimensions+d] = sum;\n" +
            "    }\n" +
            "\n" +
            "    /*if(tid == 0) {\n" +
            "        for(int i=1; i < num_threads; i++) {\n" +
            "            temp_sum[0] += temp_sum[i];\n" +
            "        }\n" +
            "        c_means[c*num_dimensions+d] = temp_sum[0];\n" +
            "        // c_means[c*num_dimensions+d] = temp_sum[0] / N[c];\n" +
            "    }*/\n" +
            "}";

    static String mstep_N = "\n" +
            "#define NUM_THREADS_MSTEP 256 // should be a power of 2 for parallel reductions to work\n" +
            "\n" +
            "__device__ float parallelSum(float* data, const unsigned int ndata) {\n" +
            "  const unsigned int tid = threadIdx.x;\n" +
            "  float t;\n" +
            "\n" +
            "  __syncthreads();\n" +
            "\n" +
            "  // Butterfly sum.  ndata MUST be a power of 2.\n" +
            "  for(unsigned int bit = ndata >> 1; bit > 0; bit >>= 1) {\n" +
            "    t = data[tid] + data[tid^bit];  __syncthreads();\n" +
            "    data[tid] = t;                  __syncthreads();\n" +
            "  }\n" +
            "  return data[tid];\n" +
            "}\n" +
            "__global__ void mstep_N(float* c_memberships, float* c_N, float* c_pi, int num_dimensions, int num_clusters, int num_events) {\n" +
            "    \n" +
            "    int tid = threadIdx.x;\n" +
            "    int num_threads = blockDim.x;\n" +
            "    int c = blockIdx.x;\n" +
            " \n" +
            "    \n" +
            "    // Need to store the sum computed by each thread so in the end\n" +
            "    // a single thread can reduce to get the final sum\n" +
            "    __shared__ float temp_sums[NUM_THREADS_MSTEP];\n" +
            "\n" +
            "    // Compute new N\n" +
            "    float sum = 0.0f;\n" +
            "    // Break all the events accross the threads, add up probabilities\n" +
            "    for(int event=tid; event < num_events; event += num_threads) {\n" +
            "        sum += c_memberships[c*num_events+event];\n" +
            "    }\n" +
            "    temp_sums[tid] = sum;\n" +
            " \n" +
            "    __syncthreads();\n" +
            "\n" +
            "    sum = parallelSum(temp_sums,NUM_THREADS_MSTEP);\n" +
            "    if(tid == 0) {\n" +
            "        c_N[c] = sum;\n" +
            "        c_pi[c] = sum;\n" +
            "    }\n" +
            "    \n" +
            "    // Let the first thread add up all the intermediate sums\n" +
            "    // Could do a parallel reduction...doubt it's really worth it for so few elements though\n" +
            "    /*if(tid == 0) {\n" +
            "        c_N[c] = 0.0;\n" +
            "        for(int j=0; j<num_threads; j++) {\n" +
            "            c_N[c] += temp_sums[j];\n" +
            "        }\n" +
            "        //printf(\"clusters[%d].N = %f\\n\",c,clusters[c].N);\n" +
            "        \n" +
            "        // Set PI to the # of expected items, and then normalize it later\n" +
            "        c_pi[c] = c_N[c];\n" +
            "    }*/\n" +
            "}";

    static String mstep_covariance1 = "\n" +
            "#define NUM_THREADS_MSTEP 256 // should be a power of 2 for parallel reductions to work\n" +
            "#define DIAG_ONLY 0\n" +
            "\n" +
            "__device__ float parallelSum(float* data, const unsigned int ndata) {\n" +
            "  const unsigned int tid = threadIdx.x;\n" +
            "  float t;\n" +
            "\n" +
            "  __syncthreads();\n" +
            "\n" +
            "  // Butterfly sum.  ndata MUST be a power of 2.\n" +
            "  for(unsigned int bit = ndata >> 1; bit > 0; bit >>= 1) {\n" +
            "    t = data[tid] + data[tid^bit];  __syncthreads();\n" +
            "    data[tid] = t;                  __syncthreads();\n" +
            "  }\n" +
            "  return data[tid];\n" +
            "}\n" +
            "\n" +
            "__device__ void compute_row_col(int n, int* row, int* col) {\n" +
            "    int i = 0;\n" +
            "    for(int r=0; r < n; r++) {\n" +
            "        for(int c=0; c <= r; c++) {\n" +
            "            if(i == blockIdx.y) {  \n" +
            "                *row = r;\n" +
            "                *col = c;\n" +
            "                return;\n" +
            "            }\n" +
            "            i++;\n" +
            "        }\n" +
            "    }\n" +
            "}\n" +
            "\n" +
            "__global__ void mstep_covariance1(float* fcs_data, float* c_R, float* c_means, float* c_memberships, float* c_avgvar, int num_dimensions, int num_clusters, int num_events) {\n" +
            "    int tid = threadIdx.x; // easier variable name for our thread ID\n" +
            "\n" +
            "    // Determine what row,col this matrix is handling, also handles the symmetric element\n" +
            "    int row,col,c;\n" +
            "    compute_row_col(num_dimensions, &row, &col);\n" +
            "    //row = blockIdx.y / num_dimensions;\n" +
            "    //col = blockIdx.y % num_dimensions;\n" +
            "\n" +
            "    __syncthreads();\n" +
            "    \n" +
            "    c = blockIdx.x; // Determines what cluster this block is handling    \n" +
            "\n" +
            "    int matrix_index = row * num_dimensions + col;\n" +
            "\n" +
            "    #if DIAG_ONLY\n" +
            "    if(row != col) {\n" +
            "        c_R[c*num_dimensions*num_dimensions+matrix_index] = 0.0;\n" +
            "        matrix_index = col*num_dimensions+row;\n" +
            "        c_R[c*num_dimensions*num_dimensions+matrix_index] = 0.0;\n" +
            "        return;\n" +
            "    }\n" +
            "    #endif \n" +
            "\n" +
            "    // Store the means in shared memory to speed up the covariance computations\n" +
            "    __shared__ float means[NUM_DIMENSIONS];\n" +
            "    // copy the means for this cluster into shared memory\n" +
            "    if(tid < num_dimensions) {\n" +
            "        means[tid] = c_means[c*num_dimensions+tid];\n" +
            "    }\n" +
            "\n" +
            "    // Sync to wait for all params to be loaded to shared memory\n" +
            "    __syncthreads();\n" +
            "\n" +
            "    __shared__ float temp_sums[NUM_THREADS_MSTEP];\n" +
            "    \n" +
            "    float cov_sum = 0.0;\n" +
            "\n" +
            "    for(int event=tid; event < num_events; event+=NUM_THREADS_MSTEP) {\n" +
            "        cov_sum += (fcs_data[row*num_events+event]-means[row])*(fcs_data[col*num_events+event]-means[col])*c_memberships[c*num_events+event]; \n" +
            "    }\n" +
            "    temp_sums[tid] = cov_sum;\n" +
            "\n" +
            "    __syncthreads();\n" +
            "\n" +
            "    cov_sum = parallelSum(temp_sums,NUM_THREADS_MSTEP);\n" +
            "    \n" +
            "    if(tid == 0) {\n" +
            "        c_R[c*num_dimensions*num_dimensions+matrix_index] = cov_sum;\n" +
            "        // Set the symmetric value\n" +
            "        matrix_index = col*num_dimensions+row;\n" +
            "        c_R[c*num_dimensions*num_dimensions+matrix_index] = cov_sum;\n" +
            "        \n" +
            "        // Regularize matrix - adds some variance to the diagonal elements\n" +
            "        // Helps keep covariance matrix non-singular (so it can be inverted)\n" +
            "        // The amount added is scaled down based on COVARIANCE_DYNAMIC_RANGE constant defined at top of this file\n" +
            "        if(row == col) {\n" +
            "            c_R[c*num_dimensions*num_dimensions+matrix_index] += c_avgvar[c];\n" +
            "        }\n" +
            "    }\n" +
            "}";

    static String mstep_covariance2 = "\n" +
            "#define COVARIANCE_DYNAMIC_RANGE 1E6\n" +
            "#define NUM_THREADS_MSTEP 256 // should be a power of 2 for parallel reductions to work\n" +
            "#define NUM_CLUSTERS_PER_BLOCK 6\n" +
            "#define DIAG_ONLY 0\n" +
            "\n" +
            "__device__ float parallelSum(float* data, const unsigned int ndata) {\n" +
            "  const unsigned int tid = threadIdx.x;\n" +
            "  float t;\n" +
            "\n" +
            "  __syncthreads();\n" +
            "\n" +
            "  // Butterfly sum.  ndata MUST be a power of 2.\n" +
            "  for(unsigned int bit = ndata >> 1; bit > 0; bit >>= 1) {\n" +
            "    t = data[tid] + data[tid^bit];  __syncthreads();\n" +
            "    data[tid] = t;                  __syncthreads();\n" +
            "  }\n" +
            "  return data[tid];\n" +
            "}\n" +
            "\n" +
            "__device__ void compute_row_col(int n, int* row, int* col) {\n" +
            "    int i = 0;\n" +
            "    for(int r=0; r < n; r++) {\n" +
            "        for(int c=0; c <= r; c++) {\n" +
            "            if(i == blockIdx.y) {  \n" +
            "                *row = r;\n" +
            "                *col = c;\n" +
            "                return;\n" +
            "            }\n" +
            "            i++;\n" +
            "        }\n" +
            "    }\n" +
            "}\n" +
            "\n" +
            "__global__ void mstep_covariance2(float* fcs_data, float* c_R, float* c_means, float* c_memberships, float* c_avgvar, int num_dimensions, int num_clusters, int num_events) {\n" +
            "    int tid = threadIdx.x; // easier variable name for our thread ID\n" +
            "\n" +
            "    // Determine what row,col this matrix is handling, also handles the symmetric element\n" +
            "    int row,col,c1;\n" +
            "    compute_row_col(num_dimensions, &row, &col);\n" +
            "\n" +
            "    __syncthreads();\n" +
            "    \n" +
            "    c1 = blockIdx.x * NUM_CLUSTERS_PER_BLOCK; // Determines what cluster this block is handling    \n" +
            "\n" +
            "    #if DIAG_ONLY\n" +
            "    if(row != col) {\n" +
            "        c_R[c*num_dimensions*num_dimensions+row*num_dimensions+col] = 0.0f;\n" +
            "        c_R[c*num_dimensions*num_dimensions+col*num_dimensions+row] = 0.0f;\n" +
            "        return;\n" +
            "    }\n" +
            "    #endif \n" +
            "\n" +
            "    // Store the means in shared memory to speed up the covariance computations\n" +
            "    __shared__ float means_row[NUM_CLUSTERS_PER_BLOCK];\n" +
            "    __shared__ float means_col[NUM_CLUSTERS_PER_BLOCK];\n" +
            "    // copy the means for this cluster into shared memory\n" +
            "    if(tid < NUM_CLUSTERS_PER_BLOCK) {\n" +
            "        means_row[tid] = c_means[(c1+tid)*num_dimensions+row];\n" +
            "        means_col[tid] = c_means[(c1+tid)*num_dimensions+col];\n" +
            "    }\n" +
            "\n" +
            "    // Sync to wait for all params to be loaded to shared memory\n" +
            "    __syncthreads();\n" +
            "\n" +
            "    __shared__ float temp_sums[NUM_THREADS_MSTEP*NUM_CLUSTERS_PER_BLOCK];\n" +
            "    \n" +
            "    float cov_sum1 = 0.0f;\n" +
            "    float cov_sum2 = 0.0f;\n" +
            "    float cov_sum3 = 0.0f;\n" +
            "    float cov_sum4 = 0.0f;\n" +
            "    float cov_sum5 = 0.0f;\n" +
            "    float cov_sum6 = 0.0f;\n" +
            "    float val1,val2;\n" +
            "        \n" +
            "    for(int c=0; c < NUM_CLUSTERS_PER_BLOCK; c++) {\n" +
            "        temp_sums[c*NUM_THREADS_MSTEP+tid] = 0.0;\n" +
            "    } \n" +
            "\n" +
            "    for(int event=tid; event < num_events; event+=NUM_THREADS_MSTEP) {\n" +
            "        val1 = fcs_data[row*num_events+event];\n" +
            "        val2 = fcs_data[col*num_events+event];\n" +
            "        cov_sum1 += (val1-means_row[0])*(val2-means_col[0])*c_memberships[c1*num_events+event]; \n" +
            "        cov_sum2 += (val1-means_row[1])*(val2-means_col[1])*c_memberships[(c1+1)*num_events+event]; \n" +
            "        cov_sum3 += (val1-means_row[2])*(val2-means_col[2])*c_memberships[(c1+2)*num_events+event]; \n" +
            "        cov_sum4 += (val1-means_row[3])*(val2-means_col[3])*c_memberships[(c1+3)*num_events+event]; \n" +
            "        cov_sum5 += (val1-means_row[4])*(val2-means_col[4])*c_memberships[(c1+4)*num_events+event]; \n" +
            "        cov_sum6 += (val1-means_row[5])*(val2-means_col[5])*c_memberships[(c1+5)*num_events+event]; \n" +
            "    }\n" +
            "    temp_sums[0*NUM_THREADS_MSTEP+tid] = cov_sum1;\n" +
            "    temp_sums[1*NUM_THREADS_MSTEP+tid] = cov_sum2;\n" +
            "    temp_sums[2*NUM_THREADS_MSTEP+tid] = cov_sum3;\n" +
            "    temp_sums[3*NUM_THREADS_MSTEP+tid] = cov_sum4;\n" +
            "    temp_sums[4*NUM_THREADS_MSTEP+tid] = cov_sum5;\n" +
            "    temp_sums[5*NUM_THREADS_MSTEP+tid] = cov_sum6;\n" +
            "\n" +
            "    __syncthreads();\n" +
            "   \n" +
            "    for(int c=0; c < NUM_CLUSTERS_PER_BLOCK; c++) {\n" +
            "        temp_sums[c*NUM_THREADS_MSTEP+tid] = parallelSum(&temp_sums[c*NUM_THREADS_MSTEP],NUM_THREADS_MSTEP);\n" +
            "        __syncthreads();\n" +
            "    }\n" +
            "    \n" +
            "    if(tid == 0) {\n" +
            "        for(int c=0; c < NUM_CLUSTERS_PER_BLOCK && (c+c1) < num_clusters; c++) {\n" +
            "            int offset = (c+c1)*num_dimensions*num_dimensions;\n" +
            "            cov_sum1 = temp_sums[c*NUM_THREADS_MSTEP];\n" +
            "            c_R[offset+row*num_dimensions+col] = cov_sum1;\n" +
            "            // Set the symmetric value\n" +
            "            c_R[offset+col*num_dimensions+row] = cov_sum1;\n" +
            "            \n" +
            "            // Regularize matrix - adds some variance to the diagonal elements\n" +
            "            // Helps keep covariance matrix non-singular (so it can be inverted)\n" +
            "            // The amount added is scaled down based on COVARIANCE_DYNAMIC_RANGE constant defined in gaussian.h\n" +
            "            if(row == col) {\n" +
            "                c_R[offset+row*num_dimensions+col] += c_avgvar[c+c1];\n" +
            "            }\n" +
            "        }\n" +
            "    }\n" +
            "}";
}
