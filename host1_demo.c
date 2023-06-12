
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_SOURCE_SIZE (0x100000)

int main(int argc,char** argv){

    // 10 work items mimic 10 processes
    int realSum=0;
    int SIZE = 10;
    int* sendCounts=(int*)malloc(sizeof(int) * SIZE);
    int* localSum=(int*)malloc(sizeof(int) * SIZE);
    int* displacements=(int*)malloc(sizeof(int) * SIZE);
    srand(time(NULL));
    int globalBufferSize = (rand()%(9000+1))+1000;
    int* globalBuffer=(int*)malloc(sizeof(int) * globalBufferSize);
    printf("The size of global array is: %d\n",globalBufferSize);

    //Initializing the globalBuffer and the localSum
    srand(time(NULL));
    for(int i=0;i<globalBufferSize;i++){
        globalBuffer[i]=(rand()%500)+1;
        realSum+=globalBuffer[i];
    }
    for(int j=0;j<SIZE;j++){
        localSum[j]=0;
    }

    //Generating variable size scatter counts for Work Items
    int temp=0;
    srand(time(NULL));
    for(int i=0;i<SIZE-1;i++){
        sendCounts[i]=(rand()%(globalBufferSize/10))+1;
        temp+=sendCounts[i];
    }
    sendCounts[SIZE-1]=(globalBufferSize-temp);
    temp=0;
    printf("The SendCounts Array is: \n");
    for(int i=0;i<SIZE;i++){
        printf("%d , ",sendCounts[i]);
        temp+=sendCounts[i];
    }
    printf("\n These SendCounts sum to %d\n",temp);

    //Generating the variable size displacements
    displacements[0]=0;
    temp=0;
    for(int i=1;i<SIZE;i++){
        displacements[i]=temp+sendCounts[i-1];
        temp+=sendCounts[i-1];
    }
    printf("The Displacements Array is: \n");
    for(int i=0;i<SIZE;i++){
        printf("%d , ",displacements[i]);
    }
    printf("\n Displacement would end at index %d\n",displacements[SIZE-1]+sendCounts[SIZE-1]);

    // Load the kernel from file kernel1.cl
	FILE* kernelFile;
	char* kernelSource;
	size_t kernelSize;
	kernelFile = fopen("kernel1_demo.cl", "r");
	if (!kernelFile) {
		fprintf(stderr, "No file named kernel1.cl was found\n");
		exit(-1);
	}
	kernelSource = (char*)malloc(MAX_SOURCE_SIZE);
	kernelSize = fread(kernelSource, 1, MAX_SOURCE_SIZE, kernelFile);
	fclose(kernelFile);

    // Getting platform and device information
	cl_platform_id platformId = NULL;
	cl_device_id deviceID = NULL;
	cl_uint retNumDevices;
	cl_uint retNumPlatforms;
	cl_int ret = clGetPlatformIDs(1, &platformId, &retNumPlatforms);
	char* value;
	size_t valueSize;
	ret = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceID, &retNumDevices);
	clGetDeviceInfo(deviceID, CL_DEVICE_NAME, 0, NULL, &valueSize);
	value = (char*)malloc(valueSize);
	clGetDeviceInfo(deviceID, CL_DEVICE_NAME, valueSize, value, NULL);
	printf("Device: %s\n", value);
	free(value);

    // Creating context.
	cl_context context = clCreateContext(NULL, 1, &deviceID, NULL, NULL, &ret);
	// Creating command queue
	cl_command_queue commandQueue = clCreateCommandQueue(context, deviceID, 0, &ret);

    // Creating Device Buffers
    cl_mem gMemObj = clCreateBuffer(context, CL_MEM_READ_ONLY, globalBufferSize * sizeof(int), NULL, &ret);
	cl_mem scMemObj = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE * sizeof(int), NULL, &ret);
	cl_mem dpMemObj = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE * sizeof(int), NULL, &ret);
    cl_mem lsMemObj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SIZE * sizeof(int), NULL, &ret);

    // Copy Arrays into the Buffers
    ret = clEnqueueWriteBuffer(commandQueue, gMemObj, CL_TRUE, 0, globalBufferSize * sizeof(int), globalBuffer, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(commandQueue, scMemObj, CL_TRUE, 0, SIZE * sizeof(int), sendCounts, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(commandQueue, dpMemObj, CL_TRUE, 0, SIZE * sizeof(int), displacements, 0, NULL, NULL);

    // Create program from kernel source
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, (const size_t*)&kernelSize, &ret);

	// Build program
	ret = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);

	// Create kernel
	cl_kernel kernel = clCreateKernel(program, "addSums", &ret);

    // Add Kernel Parameters
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&scMemObj);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&dpMemObj);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&lsMemObj);
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&gMemObj);

    // Execute the kernel
	size_t globalItemSize = SIZE;
	size_t localItemSize = 1; // globalItemSize has to be a multiple of localItemSize. 10/1 = 10 
	ret = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, NULL);

	// Read from device back to host.
	ret = clEnqueueReadBuffer(commandQueue, lsMemObj, CL_TRUE, 0, SIZE * sizeof(float), localSum, 0, NULL, NULL);

    int calcSum=0;

    for(int i=0;i<SIZE;i++){
        calcSum+=localSum[i];
    }

    // printf("Real Sum was: %d\n",realSum);
    printf("Calculated Sum was: %d\n",calcSum);

    return 0;
}