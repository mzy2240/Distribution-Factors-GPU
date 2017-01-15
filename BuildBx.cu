//kernel

__global__ void BuildBx( double * Bx, const int * frombus, const int * tobus, const int * BranchStatus, const double * xline, const int numline, const int Mat4_Width)
{
    int Row = blockIdx.y*blockDim.y+threadIdx.y;
	int Col = blockIdx.x*blockDim.x+threadIdx.x;
    
    if(Row == frombus[Row] && Col == tobus[Row])
    {
            Bx[frombus[Row]*Mat4_Width + tobus[Row]] = 1/xline[Row];
    }
}
        