% this code builds the LODF and PTDF matrices
% PTDF - POWER TRANSFER DISTRIBUTION FACTORS
% LODF - LINE OTAGE DISTRIBUTION FACTORS

PTDF = zeros(numline,numline);    % PTDF matrix
LODF = zeros(numline,numline);    % LODF matrix 
RadialLines = zeros(1,numline);   % table of lines shows radials
Bx = zeros(numbus,numbus);        % Bx matrix
Bd = zeros(numline,numline);      % diagonal matrix only
A = zeros(numline,numbus);        % line incidence matrix
flow = zeros(numline,numbus);     % line flow matrix


for iline = 1 : numline
 if BranchStatus(iline) == 1
      i = frombus(iline);
      j = tobus(iline);
      flow( iline, i ) =  1.0/xline(iline);
      flow( iline, j ) = -1.0/xline(iline);
 end
end

% build Bx matrix
for iline = 1 : numline
 if BranchStatus(iline) == 1
  Bx( frombus(iline), tobus(iline) )   =  Bx( frombus(iline), tobus(iline) )   - 1/xline(iline);
  Bx( tobus(iline),   frombus(iline) ) =  Bx( tobus(iline),   frombus(iline) ) - 1/xline(iline);
  Bx( frombus(iline), frombus(iline) ) =  Bx( frombus(iline), frombus(iline) ) + 1/xline(iline);
  Bx( tobus(iline),   tobus(iline) )   =  Bx( tobus(iline),   tobus(iline) )   + 1/xline(iline);
 end
end

B = Bx;

%Bx(refbus,refbus) = Bx(refbus,refbus) + 10000000. ; % old 1
%Bx(refbus,refbus) = Bx(refbus,refbus) + 0.0000001;  % old 2

% zero row and col for refbus, then put 1 in diag so we can invert it
Bx(:,refbus) = zeros(numbus,1);
Bx(refbus,:) = zeros(1,numbus);
Bx(refbus,refbus) = 1.0;

% get X matrix for use in DC Power Flows
if (size(Bx,1) < 32)
    I = eye(size(Bx,1));
    GridSize_Bx = ceil((size(Bx,1)*size(Bx,2))/1024) ;
    InverseKernel_Bx = parallel.gpu.CUDAKernel('CUDAInverse.ptx', 'CUDAInverse.cu', 'invert');
    InverseKernel_Bx.GridSize = [GridSize_Bx GridSize_Bx 1];
    InverseKernel_Bx.ThreadBlockSize = [32 32 1];
    d_Bx = gpuArray(double(Bx));
    d_I = gpuArray(double(I));
    N = gpuArray(int32(size(Bx,1)));
    L = zeros(size(Bx,1),size(Bx,2));
    d_Results = gpuArray(L);
    tic
    d_Results = feval(InverseKernel_Bx, d_I, d_Bx, N);
    GPU_Inv1 = toc
    Xmatrix = gather(d_Results);
    disp('IF part executed')
else
    Xmatrix = inv(Bx);
    disp('ELSE part executed')
end 

Xmatrix(refbus,refbus)=0; % set the diagonal at the ref bus to zero for a short to ground
Xmatrix;

for iline = 1 : numline
    if BranchStatus(iline) == 1
      i = frombus(iline);
      j = tobus(iline);
      Bd(iline,iline) = 1.0/xline(iline);
      A(iline,i) =  1.0;
      A(iline,j) = -1.0;
    end
end

B_diag = Bd;

%Determine Radial Lines

A_Reshape = reshape(A',1,size(A,1)*size(A,2));
A_Trans_Reshape = reshape(A,1,size(A,1)*size(A,2));
MatrixMulKernel = parallel.gpu.CUDAKernel('CUDAMul.ptx', 'CUDAMul.cu', 'MatrixMulKernel');
GridSize = ceil((size(A,1)*size(A,2))/1024);
MatrixMulKernel.ThreadBlockSize = [32 32 1];
MatrixMulKernel.GridSize = [GridSize GridSize 1];
Z = zeros(size(A',1),size(A,2));
in1 = gpuArray(A_Trans_Reshape);
in2 = gpuArray(A_Reshape);
out = gpuArray(Z);

tic
result = feval(MatrixMulKernel, out, in1, in2, size(A',2), size(A',1) , size(A,2), size(A,1), size(Z,2));
GPU_2 = toc

NumberOfLines_matrix = gather(result);
NumberOfLines = diag(NumberOfLines_matrix);
radial_bus_location = [];
radial_bus_location = find(NumberOfLines==1);
radial_bus_location

num_radialline = 0;
for n=1:length(radial_bus_location)
radial_bus = radial_bus_location(n);
        for iline = 1:numline
            if BranchStatus(iline) == 1
                if radial_bus == frombus(iline)
                    num_radialline = num_radialline + 1;
                    %RadialLines(num_radialline) = iline;
                    RadialLines(iline) = 1;
                end
            end
        end
end

for n=1:length(radial_bus_location)
radial_bus = radial_bus_location(n);
        for iline = 1:numline
            if BranchStatus(iline) == 1
                if radial_bus == tobus(iline)
                    num_radialline = num_radialline + 1;
                    %RadialLines(num_radialline) = iline;
                    RadialLines(iline) = 1;
                end
            end
        end
end
%RadialLines
%RadialLines
line_location_connecting_radial_bus = [];
line_location_connecting_radial_bus  = find(RadialLines==1);

% alter A and Bx to reflect radial lines, used only in LODF calculations
A_alt = A;
Bx_alt = Bx;

%Create A_alt matrix to account for radial lines
for iline = 1:numline
    if BranchStatus(iline) == 1
        if RadialLines(iline) == 1
            radial_bus = radial_bus_location(find(iline == line_location_connecting_radial_bus));
            A_alt(iline,radial_bus) = 0;
        end
    end
end

%Create Bx_alt matrix to account for radial lines
for ibus = 1:numbus
    if NumberOfLines(ibus) == 0 | ibus == refbus
        for jbus = 1:numbus
            Bx_alt(ibus,jbus) = 0;
        end
        Bx_alt(ibus,ibus) = 1;
    end
end

if (size(Bx_alt,1) < 32)
    I_Bx_alt = eye(size(Bx_alt,1));
    GridSize_Bx = ceil((size(Bx_alt,1)*size(Bx_alt,2))/1024) ;
    InverseKernel_Bx_alt = parallel.gpu.CUDAKernel('CUDAInverse.ptx', 'CUDAInverse.cu', 'invert');
    InverseKernel_Bx_alt.GridSize = [GridSize_Bx GridSize_Bx 1];
    InverseKernel_Bx_alt.ThreadBlockSize = [32 32 1];
    d_Bx_alt = gpuArray(double(Bx_alt));
    d_I_Bx_alt = gpuArray(double(I_Bx_alt));
    N = gpuArray(int32(size(Bx_alt,1)));
    L_Bx_alt = zeros(size(Bx_alt,1),size(Bx_alt,2));
    d_Results_Bx_alt = gpuArray(L_Bx_alt);
    tic
    d_Results_Bx_alt = feval(InverseKernel_Bx_alt, d_I_Bx_alt, d_Bx_alt, N);
    GPU_Inv2 = toc
    
    X_alt = gather(d_Results_Bx_alt);
 
else
    X_alt = inv(Bx_alt);
end
    
X_alt(refbus,refbus)=0; % set the diagonal at the ref bus to zero for a short to ground


% basic expression for PTDF matrix which includes the PTDF(K,K) on
% diagonals and is compensated for radial lines.

% B_diag * A_alt
B_diag_Reshape = reshape(B_diag',1,size(B_diag,1)*size(B_diag,2));
A_alt_Reshape = reshape(A_alt',1,size(A_alt,1)*size(A_alt,2));
MatrixMulKernel2 = parallel.gpu.CUDAKernel('CUDAMul.ptx', 'CUDAMul.cu', 'MatrixMulKernel');
GridSize2 = ceil((size(A_alt,1)*size(A_alt,2))/1024);
MatrixMulKernel2.ThreadBlockSize = [32 32 1];
MatrixMulKernel2.GridSize = [GridSize2 GridSize2 1];
Z1 = zeros(size(B_diag,1),size(A_alt,2));
in3 = gpuArray(B_diag_Reshape);
in4 = gpuArray(A_alt_Reshape);
out2 = gpuArray(Z1);
tic 
result2 = feval(MatrixMulKernel2, out2, in3, in4, size(B_diag,2), size(B_diag,1) , size(A_alt,2), size(A_alt,1), size(Z1,1));
GPU_4 = toc
PTDF1_CUDA = gather(result2);
clear in3
clear in4



% X_alt * A_alt'
X_alt_Reshape = reshape(X_alt',1,size(X_alt,1)*size(X_alt,2));
A_alt_Trans_Reshape = reshape(A_alt,1,size(A_alt',1)*size(A_alt',2));
MatrixMulKernel3 = parallel.gpu.CUDAKernel('CUDAMul.ptx', 'CUDAMul.cu', 'MatrixMulKernel');
GridSize3 = ceil((size(X_alt,1)*size(X_alt,2))/1024);
MatrixMulKernel3.ThreadBlockSize = [32 32 1];
MatrixMulKernel3.GridSize = [GridSize3 GridSize3 1];
Z2 = zeros(size(X_alt,1),size(A_alt',2));
in5 = gpuArray(X_alt_Reshape);
in6 = gpuArray(A_alt_Trans_Reshape);
out2 = gpuArray(Z2);
tic
result3 = feval(MatrixMulKernel3, out2, in5, in6, size(X_alt,2), size(X_alt,1) , size(A_alt',2), size(A_alt',1), size(Z2,1));
GPU_5 = toc



PTDF2_CUDA = gather(result3);

% PTDF1_CUDA * PTDF2_CUDA
PTDF1_CUDA_Reshape = reshape(PTDF1_CUDA',1,size(PTDF1_CUDA,1)*size(PTDF1_CUDA,2));
PTDF2_CUDA_Reshape = reshape(PTDF2_CUDA',1,size(PTDF2_CUDA,1)*size(PTDF2_CUDA,2));
MatrixMulKernel4 = parallel.gpu.CUDAKernel('CUDAMul.ptx', 'CUDAMul.cu', 'MatrixMulKernel');
GridSize4 = ceil((size(PTDF1_CUDA,1)*size(PTDF1_CUDA,2))/1024);
MatrixMulKernel4.ThreadBlockSize = [32 32 1];
MatrixMulKernel4.GridSize = [GridSize4 GridSize4 1];
Z4 = zeros(size(PTDF1_CUDA,1),size(PTDF2_CUDA,2));
in5 = gpuArray(PTDF1_CUDA_Reshape);
in6 = gpuArray(PTDF2_CUDA_Reshape);
out4 = gpuArray(Z4);
tic
result4 = feval(MatrixMulKernel4, out4, in5, in6, size(PTDF1_CUDA,2), size(PTDF1_CUDA,1) , size(PTDF2_CUDA,2), size(PTDF2_CUDA,1), size(Z4,1));
GPU_6 = toc
PTDF = gather(result4);

clear in5
clear in6

% set PTDF diagonal to zero for radial lines
for iline = 1:numline
    if RadialLines(iline) == 1
        PTDF(iline,iline) = 0;
    end
end
PTDF;

% LODF(L,K) (or dfactor) = PTDF(L,K) / (1 - PTDF(K,K) ) 

% First we need to check to see that a line outage will not cause islanding
% this is detected when the diagonal of any line in PTDF is very close to
% 1.0. In this case if such a line is detected, we force the PTDF(K,K) to
% zero so that we do not get a divide by zero and issue an error warning of
% islanding.

PTFD_denominator = PTDF ;
%diag(PTFD_denominator)

for iline = 1:numline
        if (1.0 - PTFD_denominator(iline,iline) ) < 1.0E-06
            PTFD_denominator(iline,iline) = 0.0 ;
            fprintf(' Loss of line from %3d to %3d will cause islanding \n',frombus(iline), tobus(iline));
        end
end

% diag(PTDF) extracts the diagonals of PTDF matrix into a vector
% diag(diag(PTDF)) extracts diags of PTDF matrix and put them into a matrix
% of the same size with all zeros in off diagonals.
% expression below multiplies the PTDF matrix by a matrix with diagonals
% equal to 1/(1 - PTDF(K,K))

LODF_Part1 = (speye(numline)-diag(diag(PTFD_denominator)));

if (size(LODF_Part1,1) < 32)
    I_LODF_Part1 = eye(size(LODF_Part1,1));
    GridSize_LODF_Part1 = ceil((size(LODF_Part1,1)*size(LODF_Part1,2))/1024) ;
    InverseKernel_LODF_Part1 = parallel.gpu.CUDAKernel('CUDAInverse.ptx', 'CUDAInverse.cu', 'invert');
    InverseKernel_LODF_Part1.GridSize = [GridSize_LODF_Part1 GridSize_LODF_Part1 1];
    InverseKernel_LODF_Part1.ThreadBlockSize = [32 32 1];
    d_LODF_Part1 = gpuArray(double(LODF_Part1));
    d_I_LODF_Part1 = gpuArray(double(I_LODF_Part1));
    N = gpuArray(int32(size(LODF_Part1,1)));
    L_LODF_Part1 = zeros(size(LODF_Part1,1),size(LODF_Part1,2));
    d_Results_LODF_Part1 = gpuArray(L_LODF_Part1);
    tic
    d_Results_LODF_Part1 = feval(InverseKernel_LODF_Part1, d_I_LODF_Part1, d_LODF_Part1, N);
    toc
    LODF_Part2 = gather(d_Results_LODF_Part1);
else
    LODF_Part2 = inv(LODF_Part1);
end

%LODF = PTDF* MatrixInverse( speye(numline)-diag(diag(PTFD_denominator)) );

LODF_Part2_Reshape = reshape(LODF_Part2',1,size(LODF_Part2,1)*size(LODF_Part2,2));
PTDF_Reshape = reshape(PTDF',1,size(PTDF,1)*size(PTDF,2));
MatrixMulKernel5 = parallel.gpu.CUDAKernel('CUDAMul.ptx', 'CUDAMul.cu', 'MatrixMulKernel');
GridSize5 = ceil((size(PTDF,1)*size(PTDF,2))/1024);
MatrixMulKernel5.ThreadBlockSize = [32 32 1];
MatrixMulKernel5.GridSize = [GridSize5 GridSize5 1];
Z5 = zeros(size(PTDF,1),size(LODF_Part2,2));
in7 = gpuArray(PTDF_Reshape);
in8 = gpuArray(LODF_Part2_Reshape);
out5 = gpuArray(Z5);
tic
result5 = feval(MatrixMulKernel5, out5, in7, in8, size(PTDF,2), size(PTDF,1) , size(LODF_Part2,2), size(LODF_Part2,1), size(Z5,1));
GPU_1 = toc 
LODF = gather(result5);

for iline = 1:numline
      LODF(iline,iline) = 0;
end
   
LODF;

if printfactorsflag == 1
    
    %--------------------------------------------------------------------
    %--------------------------------------------------------------------

    % Calculate the single injection to line flow factor matrix
    % call this the afact matrix. Assumes injections are positive and
    % compensated by an equal negative drop on the reference bus

    Bx2 = B;
    Bx2(refbus,refbus) = Bx2(refbus,refbus) + 10000000. ; % makes matrix non singular

    % inverse 
    if (size(Bx2,1) < 32)
        I_Bx2 = eye(size(Bx2,1));
        GridSize_Bx2 = ceil((size(Bx2,1)*size(Bx2,2))/1024) ;
        InverseKernel_Bx2 = parallel.gpu.CUDAKernel('CUDAInverse.ptx', 'CUDAInverse.cu', 'invert');
        InverseKernel_Bx2.GridSize = [GridSize_Bx2 GridSize_Bx2 1] ;
        InverseKernel_Bx2.ThreadBlockSize = [32 32 1];
        d_Bx2 = gpuArray(double(Bx2));
        d_I_Bx2 = gpuArray(double(I_Bx2)) ;
        N = gpuArray(int32(size(Bx2,1))) ;
        L_Bx2 = zeros(size(Bx_alt,1),size(Bx_alt,2)) ;
        d_Results_Bx2 = gpuArray(L_Bx2);
        d_Results_Bx2 = feval(InverseKernel_Bx2, d_I_Bx2, d_Bx2, N);
        Xmatrix = gather(d_Results_Bx2);
    else
        Xmatrix = inv(Bx2);
    end

    % loop on the monitored line imon from i to j
    for imon = 1 : numline

         i = frombus(imon);
         j = tobus(imon);

        % loop on injection bus s
        for s = 1 : numbus
         if s ~= refbus
            afact(imon,s) = (1/xline(imon))*(Xmatrix(i,s) - Xmatrix(j,s));
         else
            afact(imon,s) = 0.0;
         end
        end
    end
    
    GPU_Time = GPU_1 + GPU_2 + GPU_4 + GPU_5 + GPU_6 
    
    fprintf('%s\n','AFACT MATRIX');
    fprintf('%s\n','Monitored      GENERATOR');
    fprintf('%s\n','Line           ');
    fprintf('\n');
    fprintf('%s','              ');
      for s = 1 : numbus
          fprintf('%s %2d %s','  ',s,'      ');
      end
    fprintf('\n');
    fprintf('\n');


    for imon = 1 : numline 
        fprintf('%2d %s %2d %s',frombus(imon),'to', tobus(imon),'    ');
        for s = 1 : numbus
            fprintf('%8.4f %s',afact(imon,s),'   ');
        end
        fprintf('\n');
    end
   
    %--------------------------------------------------------------------
    fprintf('\n');
    fprintf('\n');
    fprintf('%s\n','POWER TRANSFER DISTRIBUTION FACTOR (PTDF) MATRIX');
    fprintf('%s\n','Monitored      Transaction');
    fprintf('%s\n','Line           From(Sell) - To(Buy)');
    fprintf('\n');
    fprintf('%s','              ');
    for t = 1 : numline
        fprintf('%2d %s %2d %s',frombus(t),'to',tobus(t),'   ');
    end
    fprintf('\n');
    fprintf('\n');


    for imon = 1 : numline 
        fprintf('%2d %s %2d %s',frombus(imon),'to', tobus(imon),'    ');
        for t = 1 : numline
           fprintf('%8.4f %s',PTDF(imon,t),'   ');
        end
        fprintf('\n');
    end
    
    %--------------------------------------------------------------------
    fprintf('\n');
    fprintf('\n');
    fprintf('%s\n','LINE OUTAGE DISTRIBUTION FACTOR (LODF) MATRIX');
    fprintf('%s\n','Monitored      Outage of one circuit');
    fprintf('%s\n','Line           From - To');
    fprintf('\n');
    fprintf('%s','              ');
    for idrop = 1 : numline
        fprintf('%2d %s %2d %s',frombus(idrop),'to',tobus(idrop),'   ');
    end
    fprintf('\n');
    fprintf('\n');


    for imon = 1 : numline 
       fprintf('%2d %s %2d %s',frombus(imon),'to', tobus(imon),'    ');
           for idrop = 1 : numline
                fprintf('%8.4f %s',LODF(imon, idrop),'   ');
           end
           fprintf('\n');
    end
    fprintf('\n');
    fprintf('\n');
    
    
end

