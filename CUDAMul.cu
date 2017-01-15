// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel( double * P ,const double * M, const double * N , const int Mat1_Width, const int Mat1_Height, const int Mat2_Width, const int Mat2_Height , const int Mat3_Width)
{
    int Row = blockIdx.y*blockDim.y+threadIdx.y;
	int Col = blockIdx.x*blockDim.x+threadIdx.x;

    double Pvalue = 0.0;

    if((Row < Mat1_Height) && (Col < Mat2_Width))
    {
        
        for (int k = 0; k < Mat1_Width ; ++k)
           {
              Pvalue += M[Row*Mat1_Width + k] * N[k*Mat2_Width + Col];
           }
    
        P[Col*Mat3_Width + Row] = Pvalue;

    }   
}