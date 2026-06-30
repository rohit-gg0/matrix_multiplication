#include<cuda_runtime.h>
#include<math.h>
#include<float.h>
#include<curand.h>
#include<stdio.h>
#include<stdlib.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define BATCH_SIZE 128

void init_parameters(float* weight1,float* bias1,float* weight2,float* bias2)
{
    float w1 = sqrtf(6.0f/(INPUT_SIZE+HIDDEN_SIZE));
    float w2 = sqrtf(6.0f/(HIDDEN_SIZE+OUTPUT_SIZE));

    for(int i=0;i<INPUT_SIZE*HIDDEN_SIZE;i++)
    {
        weight1[i] = (2.0f*(float)rand()/RAND_MAX - 1.0f)*w1;
    }
    for(int i=0;i<HIDDEN_SIZE;i++)
    {
        bias1[i]=0.0f;
    }

    for(int i=0;i<HIDDEN_SIZE*OUTPUT_SIZE;i++)
    {
        weight2[i] = (2.0f*(float)rand()/RAND_MAX - 1.0f)*w2;
    }
    for(int i=0;i<OUTPUT_SIZE;i++)
    {
        bias2[i]=0.0f;
    }
}

int reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void loadMNISTImages(const char* filename, float* images, int num_images)
{
    FILE* file = fopen(filename, "rb");

    if (!file) {
        printf("Cannot open image file!\n");
        exit(1);
    }

    int magic_number = 0;
    int number_of_images = 0;
    int rows = 0;
    int cols = 0;

    fread(&magic_number, sizeof(magic_number), 1, file);
    magic_number = reverseInt(magic_number);

    fread(&number_of_images, sizeof(number_of_images), 1, file);
    number_of_images = reverseInt(number_of_images);

    fread(&rows, sizeof(rows), 1, file);
    rows = reverseInt(rows);

    fread(&cols, sizeof(cols), 1, file);
    cols = reverseInt(cols);

    printf("Images: %d\n", number_of_images);
    printf("Rows: %d Cols: %d\n", rows, cols);

    for (int i = 0; i < num_images; i++) {
        for (int j = 0; j < rows * cols; j++) {

            unsigned char temp = 0;
            fread(&temp, sizeof(temp), 1, file);

            // Normalize to [0,1]
            images[i * rows * cols + j] =
                ((float)temp) / 255.0f;
        }
    }

    fclose(file);
}

void loadMNISTLabels(const char* filename, int* labels, int num_labels)
{
    FILE* file = fopen(filename, "rb");

    if (!file) {
        printf("Cannot open label file!\n");
        exit(1);
    }

    int magic_number = 0;
    int number_of_labels = 0;

    fread(&magic_number, sizeof(magic_number), 1, file);
    magic_number = reverseInt(magic_number);

    fread(&number_of_labels, sizeof(number_of_labels), 1, file);
    number_of_labels = reverseInt(number_of_labels);

    printf("Labels: %d\n", number_of_labels);

    for (int i = 0; i < num_labels; i++) {

        unsigned char temp = 0;
        fread(&temp, sizeof(temp), 1, file);

        labels[i] = (int)temp;
    }

    fclose(file);
}

void get_input(float* input,int* label)
{
    loadMNISTImages("t10k-images.idx3-ubyte",input,BATCH_SIZE);
    loadMNISTLabels("t10k-labels.idx1-ubyte",label,BATCH_SIZE);
}

__global__ void matmul(float* A,float* B,float* C,int na,int ma,int mb)
{
    int i = blockDim.y*blockIdx.y + threadIdx.y;
    int j = blockDim.x*blockIdx.x + threadIdx.x;

    if(i<na && j<mb)
    {
        float sum=0.0f;
        for(int k=0;k<ma;k++)
        {
            sum+=A[i*ma+k]*B[k*mb+j];
        }
        C[i*mb+j] = sum;
    }
}

__global__ void addBias(float* output, float* bias,int n,int m)
{
    int i = blockDim.y*blockIdx.y + threadIdx.y;
    int j = blockDim.x*blockIdx.x + threadIdx.x;

    if(i<n && j<m)
    {
        output[i*m+j] += bias[j];
    }
}

__global__ void relu(float* data,int size)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<size)
    {
        data[i] = fmaxf(0.0f,data[i]);
    }
}

__global__ void softmax(float* input,float* output,int batch_size,int num_classes)
{
    int batch_idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(batch_idx<batch_size)
    {
        float max = -FLT_MAX;
        for(int i=0;i<num_classes;i++)
        {
            max = fmaxf(max,input[batch_idx*num_classes+i]);
        }

        float sum=0.0f;
        for(int i=0;i<num_classes;i++)
        {
            output[batch_idx*num_classes+i] = expf(input[batch_idx*num_classes+i]-max);
            sum+=output[batch_idx*num_classes+i];
        }

        for(int i=0;i<num_classes;i++)
        {
            output[batch_idx*num_classes+i] /= sum;
        }
    }
}

__global__ void getprediction(float* softmax_output,int* predictions,int batch_size,int num_classes)
{
    int batch_idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(batch_idx<batch_size)
    {
        float max_prob = 0.0f;
        int max_i = -1;

        for(int i=0;i<num_classes;i++)
        {
            float prob = softmax_output[batch_idx*num_classes+i];
            if(prob>max_prob)
            {
                max_prob = prob;
                max_i = i;
            }
        }

        predictions[batch_idx] = max_i;
    }
}

void forwardPass(float* d_input,float* d_weights1,float* d_bias1,float* d_weights2,float* d_bias2,float* d_hidden_preact,float* d_hidden_output,float* d_output_preact,float* d_output,int* d_predictions)
{
    dim3 block_mm(16,16);
    dim3 grid_mm1((HIDDEN_SIZE+block_mm.x-1)/block_mm.x , (BATCH_SIZE+block_mm.y-1)/block_mm.y);
    dim3 grid_mm2((OUTPUT_SIZE+block_mm.x-1)/block_mm.x , (BATCH_SIZE+block_mm.y-1)/block_mm.y);

    dim3 block_bias(16,16);
    dim3 grid_bias1((HIDDEN_SIZE+block_mm.x-1)/block_mm.x , (BATCH_SIZE+block_mm.y-1)/block_mm.y);
    dim3 grid_bias2((OUTPUT_SIZE+block_mm.x-1)/block_mm.x , (BATCH_SIZE+block_mm.y-1)/block_mm.y);

    int block_act = 256;
    int grid_act1 = (BATCH_SIZE*HIDDEN_SIZE+block_act-1)/block_act;

    int block_pred = 256;
    int grid_pred = (BATCH_SIZE+block_pred-1)/block_pred;

    matmul<<<grid_mm1,block_mm>>>(d_input,d_weights1,d_hidden_preact,BATCH_SIZE,INPUT_SIZE,HIDDEN_SIZE);

    addBias<<<grid_bias1,block_bias>>>(d_hidden_preact,d_bias1,BATCH_SIZE,HIDDEN_SIZE);

    relu<<<grid_act1,block_act>>>(d_hidden_preact,BATCH_SIZE*HIDDEN_SIZE);

    cudaMemcpy(d_hidden_output,d_hidden_preact,BATCH_SIZE*HIDDEN_SIZE*sizeof(float),cudaMemcpyDeviceToDevice);

    matmul<<<grid_mm2,block_mm>>>(d_hidden_output,d_weights2,d_output_preact,BATCH_SIZE,HIDDEN_SIZE,OUTPUT_SIZE);

    addBias<<<grid_bias2,block_bias>>>(d_output_preact,d_bias2,BATCH_SIZE,OUTPUT_SIZE);

    softmax<<<grid_pred,block_pred>>>(d_output_preact,d_output,BATCH_SIZE,OUTPUT_SIZE);

    getprediction<<<grid_pred,block_pred>>>(d_output,d_predictions,BATCH_SIZE,OUTPUT_SIZE);
}

float calc_accuracy(int *predictions, int *labels, int batch_size)
{
    int correct = 0;
    for (int i = 0; i < batch_size; i++) 
    {
        if (predictions[i] == labels[i]) 
        {
            correct++;
        }
    }
    return (float)correct / batch_size;
}

int main()
{
    srand(42);

    float* h_weights1 = (float*)malloc(INPUT_SIZE*HIDDEN_SIZE*sizeof(float));
    float* h_bias1 = (float*)malloc(HIDDEN_SIZE*sizeof(float));
    float* h_weights2 = (float*)malloc(HIDDEN_SIZE*OUTPUT_SIZE*sizeof(float));
    float* h_bias2 = (float*)malloc(OUTPUT_SIZE*sizeof(float));

    init_parameters(h_weights1,h_bias1,h_weights2,h_bias2);

    float* d_weights1;
    float* d_bias1;
    float* d_weights2;
    float* d_bias2;

    cudaMalloc(&d_weights1,INPUT_SIZE*HIDDEN_SIZE*sizeof(float));
    cudaMalloc(&d_bias1,HIDDEN_SIZE*sizeof(float));
    cudaMalloc(&d_weights2,HIDDEN_SIZE*OUTPUT_SIZE*sizeof(float));
    cudaMalloc(&d_bias2,OUTPUT_SIZE*sizeof(float));

    cudaMemcpy(d_weights1,h_weights1,INPUT_SIZE*HIDDEN_SIZE*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias1,h_bias1,HIDDEN_SIZE*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights2,h_weights2,HIDDEN_SIZE*OUTPUT_SIZE*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias2,h_bias2,OUTPUT_SIZE*sizeof(float),cudaMemcpyHostToDevice);

    float* h_input = (float*)malloc(BATCH_SIZE*INPUT_SIZE*sizeof(float));
    int* h_labels = (int*)malloc(BATCH_SIZE*sizeof(int));

    get_input(h_input,h_labels);

    float* d_input;
    cudaMalloc(&d_input,BATCH_SIZE*INPUT_SIZE*sizeof(float));
    cudaMemcpy(d_input,h_input,BATCH_SIZE*INPUT_SIZE*sizeof(float),cudaMemcpyHostToDevice);

    float* d_hidden_preact;
    float* d_hidden_output;
    float* d_output_preact;
    float* d_output;

    cudaMalloc(&d_hidden_preact,BATCH_SIZE*HIDDEN_SIZE*sizeof(float));
    cudaMalloc(&d_hidden_output,BATCH_SIZE*HIDDEN_SIZE*sizeof(float));
    cudaMalloc(&d_output_preact,BATCH_SIZE*OUTPUT_SIZE*sizeof(float));
    cudaMalloc(&d_output,BATCH_SIZE*OUTPUT_SIZE*sizeof(float));

    int* h_predictions = (int*)malloc(BATCH_SIZE*sizeof(int));
    int* d_predictions;
    cudaMalloc(&d_predictions,BATCH_SIZE*sizeof(int));

    forwardPass(d_input,d_weights1,d_bias1,d_weights2,d_bias2,d_hidden_preact,d_hidden_output,d_output_preact,d_output,d_predictions);

    cudaDeviceSynchronize();

    cudaMemcpy(h_predictions,d_predictions,BATCH_SIZE*sizeof(int),cudaMemcpyDeviceToHost);

    float* h_output = (float*)malloc(BATCH_SIZE*OUTPUT_SIZE*sizeof(float));
    cudaMemcpy(h_output,d_output,BATCH_SIZE*OUTPUT_SIZE*sizeof(float),cudaMemcpyDeviceToHost);

    float accuracy = calc_accuracy(h_predictions,h_labels,BATCH_SIZE);

    printf("\nAccuracy: %.2f%%\n",accuracy * 100.0f);

    for (int i = 0; i < BATCH_SIZE; i++) 
    {
        printf("Sample %d - True label: %d, Predicted: %d\n", i, h_labels[i], h_predictions[i]);
        printf("  Probabilities: ");
        for (int j = 0; j < OUTPUT_SIZE; j++) 
        {
            printf("%.4f ", h_output[i * OUTPUT_SIZE + j]);
        }
        printf("\n");
    }

    cudaFree(d_weights1);
    cudaFree(d_bias1);
    cudaFree(d_weights2);
    cudaFree(d_bias2);
    cudaFree(d_input);
    cudaFree(d_hidden_preact);
    cudaFree(d_hidden_output);
    cudaFree(d_output_preact);
    cudaFree(d_output);
    cudaFree(d_predictions);
    
    free(h_weights1);
    free(h_bias1);
    free(h_weights2);
    free(h_bias2);
    free(h_input);
    free(h_labels);
    free(h_predictions);
    free(h_output);
}