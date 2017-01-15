This material is the MATLAB/CUDA demo component for the webinar titled:
"MATLAB for CUDA Programmers"

To run this demo, you will need the following products from MathWorks:
	MATLAB
	Parallel Computing toolbox

In addition, you will need the following third-party tools installed:
	nVidia GPU Computing Toolkit v5
	An appropriate compiler. For Windows the following have been tested:
	-Microsoft Visual Studio 2010 Professional
	-Microsoft Visual Studio 2010 Express 
	  For Express edition, make sure you have read the following:
		http://www.mathworks.se/support/solutions/en/data/1-FOYO7U/index.html?product=DM&solution=1-FOYO7U

In each folder there are CUDA-files that need to be compiled for the 
GPU demos to work. In each folder is a MATLAB script called buildMeWin.m
	buildMeWin.m
		Compile script for Windows. Assumes nvcc is on the path and that 
		Microsoft Visual Studio 2010 (Professional or Express) is installed
		in the default folder (C:\Program Files (x86)\Microsoft Visual 
		Studio 10.0). If not, the hard-coded path has to be adjusted 
		accordingly. Will automatically determine if you are running 32-bit 
		or 64-bit MATLAB and compile accordingly. 
Run the appropriate build script and verify that there are no errors, and 
that *.ptx files are generated.

Hope it works and gives you some new ideas.

Daniel Armyr
Application Engineer
MathWorks