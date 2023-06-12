__kernel void addSums(__global const int *sc, __global const int *dp,__global int *output,const int max,__global const int *input,const int cp,const int size,__global const int *sparse,__global int* dot) {
        int gid = get_global_id(0);
        int count=sc[gid];
        int start=dp[gid];
        
        __local int val;
        
        int offset = gid*max;
        for(int i=0;i<count;i++){
            output[start+i]=input[offset+i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if(gid==cp){
            printf("Inside Kernel dotP: %d\n",cp);
            for(int i=0;i<size;i++){
                val=val+(output[i]*sparse[i]);
            }
            dot[gid]=val;
            printf("dot is %d\n",dot[gid]);
            printf("Val is %d\n",val);
        }
}