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

int main(int charc, char** charv){
    int SIZE=10;
    int dotProduct;
    int* sendCounts=(int*)malloc(SIZE*sizeof(int));
    int* displacements=(int*)malloc(SIZE*sizeof(int));
    
    int temp=0;
    int* globalBuffer;
    int globalBufferSize;
    int maxCount=0;
    int realSum=0;

    // Create Send Counts
    srand(time(NULL));
    for(int i=0;i<SIZE;i++){
        sendCounts[i]=(rand()%180)+20;
        // sendCounts[i]=(rand()%5)+2;
        if(sendCounts[i]>maxCount){
            maxCount=sendCounts[i];
        }
        temp+=sendCounts[i];
    }
    globalBufferSize=temp;
    printf("globalBufferSize is: %d\n",globalBufferSize);
    printf("Max Count is: %d\n",maxCount);

    temp=0;
    printf("The SendCounts Array is: \n");
    for(int i=0;i<SIZE;i++){
        printf("%d , ",sendCounts[i]);
        temp+=sendCounts[i];
    }
    printf("\n These SendCounts sum to %d\n",temp);

    //Create Displacements 
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

    //generate a 2D array globalBuffer SIZE X maxCount
    globalBuffer = (int *)malloc(SIZE * maxCount * sizeof(int)); 
    for(int i=0;i<SIZE;i++){
        int count = sendCounts[i];
        int k=0;
        for(int j=0;j<count;j++){
             *(globalBuffer + i*maxCount + j)=(rand()%500)+1;
             realSum+=*(globalBuffer + i*maxCount + j);
             k=j;
        }
        k++;
        for(;k<maxCount;k++){
            *(globalBuffer + i*maxCount + k)=0;
        }
    }

    // printf("\n Global Buffer: \n");

    // for(int i=0;i<SIZE;i++){
    //     for(int j=0;j<maxCount;j++){
    //         printf("%d\t",*(globalBuffer + i*maxCount + j));
    //     }
    //     printf("\n");
    // }

    //Create and initialize the Sparse Matrix
    int* sparseMatrix = (int*)malloc(globalBufferSize*sizeof(int));
    int* gatheredResult = (int*)malloc(globalBufferSize*sizeof(int));

    for(int j=0;j<globalBufferSize;j++){
        gatheredResult[j]=0;
    }

    // printf("Gathered Array: \n");
    // for(int i=0;i<globalBufferSize;i++){
    //     printf("%d\t",gatheredResult[i]);
    // }

    srand(time(NULL));
    for(int i=0;i<globalBufferSize;i++){
        sparseMatrix[i]=(rand()%10);
    }

    // generate the coordinatiing process
    srand(time(NULL));
    int coordinatingProcess = rand()%SIZE;

    // Load the kernel from file kernel1.cl
	FILE* kernelFile;
	char* kernelSource;
	size_t kernelSize;
	kernelFile = fopen("kernel2.cl", "r");
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
    // cl_mem maxMemObj = clCreateBuffer(context, CL_MEM_READ_ONLY,sizeof(int), NULL, &ret);
    cl_mem inMemObj = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE * maxCount* sizeof(int), NULL, &ret);
    cl_mem scMemObj = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE * sizeof(int), NULL, &ret);
    cl_mem dotMemObj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,SIZE*sizeof(int), NULL, &ret);
	cl_mem dpMemObj = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE * sizeof(int), NULL, &ret);
    cl_mem outMemObj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, globalBufferSize * sizeof(int), NULL, &ret);
    cl_mem smMemObj = clCreateBuffer(context, CL_MEM_READ_ONLY, globalBufferSize * sizeof(int), NULL, &ret);

    // Writing Device Buffers
    // ret = clEnqueueWriteBuffer(commandQueue, maxMemObj, CL_TRUE, 0,sizeof(int), &maxCount, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(commandQueue, inMemObj, CL_TRUE, 0, SIZE * maxCount* sizeof(int), globalBuffer, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(commandQueue, scMemObj, CL_TRUE, 0, SIZE * sizeof(int), sendCounts, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(commandQueue, dpMemObj, CL_TRUE, 0, SIZE * sizeof(int), displacements, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(commandQueue, smMemObj, CL_TRUE, 0, globalBufferSize * sizeof(int), sparseMatrix, 0, NULL, NULL);

    // Create program from kernel source
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, (const size_t*)&kernelSize, &ret);

	// Build program
	ret = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);

	// Create kernel
	cl_kernel kernel = clCreateKernel(program, "addSums", &ret);

    //Add Parameters
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&scMemObj);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&dpMemObj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&outMemObj);
    ret = clSetKernelArg(kernel, 3, sizeof(int), &maxCount);
    ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&inMemObj);
    ret = clSetKernelArg(kernel, 5, sizeof(int), &coordinatingProcess);
    ret = clSetKernelArg(kernel, 6, sizeof(int), &globalBufferSize);
    ret = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&smMemObj);
    ret = clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*)&dotMemObj);

    // Execute the kernel
	size_t globalItemSize = SIZE;
	size_t localItemSize = 1; // globalItemSize has to be a multiple of localItemSize. 10/1 = 10 
	ret = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, NULL);


    // Read from device back to host.
    int* dotps=(int*)malloc(sizeof(int)* SIZE);

	ret = clEnqueueReadBuffer(commandQueue, outMemObj, CL_TRUE, 0, globalBufferSize * sizeof(int), gatheredResult, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(commandQueue, dotMemObj, CL_TRUE, 0, SIZE* sizeof(int), dotps, 0, NULL, NULL);

    int calcSum=0;

    printf("GATHERED RESULT: \n");
    for(int i=0;i<globalBufferSize;i++){
        //printf("%d\t",gatheredResult[i]);
        calcSum+=gatheredResult[i];
    }

    printf("\nReal Sum was: %d\n",realSum);
    printf("Calculated Sum was: %d\n",calcSum);
    printf("Dot Product is: %d\n",dotps[coordinatingProcess]);

    return 0;
}