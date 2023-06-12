__kernel void addSums(__global const int *sc, __global const int *dp,__global int *ls,__global int* buf) {
        printf("HelloWorld\n");
        int gid = get_global_id(0);
		int count=sc[gid];
        int start=dp[gid];
        int sum=0;
        for(int i=start;i<start+count;i++){
            sum+=buf[i];
        }
        ls[gid]=sum;
}