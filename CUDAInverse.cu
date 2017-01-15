
__global__ void invert(double * I, double * A, const int * n){
    for (int i = 0; i<n[0]; i++){
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        //Non diagonal normalization

        if (x < n[0] && y < n[0])
        if (x == i && x!=y){
            I[x*n[0] + y] /= A[i*n[0] + i];
            A[x*n[0] + y] /= A[i*n[0] + i];
        }
        __syncthreads();

        //Diagonal normalization

        if (x < n[0] && y < n[0])
        if (x == y && x == i){
            I[x*n[0] + y] /= A[i*n[0] + i];
            A[x*n[0] + y] /= A[i*n[0] + i];
        }
        __syncthreads();

        //Gauss Jordan Elimination

        if (x < n[0] && y < n[0]){
            if (x != i){
                I[x*n[0] + y] -= I[i*n[0] + y] * A[x*n[0] + i];
                if (y != i){
                    A[x*n[0] + y] -= A[i*n[0] + y] * A[x*n[0] + i];
                }	 
            }
        }
        __syncthreads();

        //Set to zero

        if (x < n[0] && y < n[0]){
            if (x != i){
                if (y == i){
                    A[x*n[0] + y] = 0;
                }
            }
        }
    }
    for(int a = 0; a < n[0]*n[0]; a++)
        A[a] = 5.0;
}
    
