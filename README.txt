INTRODUCTION:

Distribution factors are used mainly in security and contingency
analysis. They are used to approximately determine the impact of
generation and load on transmission flows. Power Transfer
Distribution Factor (PTDF) and Load Outage Distribution factor
(LODF) as two such factors which will give an insight on effects
of power generation and load. PTDF calculates a relative change
in power flow on a particular line due to a change in injection and
corresponding withdrawal at a pair of busses and LODF
calculates a redistribution of power in the system in case of an
outage. Goal of this project is to parallelize the calculations using
CUDA C on the GPU GTX 480.

User Manual:

Project folder consists of following Excel Spreadsheet data files which are used to test the systems for different sizes of power system:
6BusCase_networkdata.xlsx
12BusCase_networkdata.xlsx
20BusCase_networkdata.xlsx
30BusCase_networkdata.xlsx
118BusCase_networkdata.xlsx
150BusCase_networkdata.xlsx
200BusCase_networkdata.xlsx
300BusCase_networkdata.xlsx
500BusCase_networkdata.xlsx
800BusCase_networkdata.xlsx
1000BusCase_networkdata.xlsx
2730BusCase_networkdata.xlsx

Before running the program, please make sure that GPU drivers are installed and configured in the system. 

There are 3 CUDA files
CUDAMul.cu
CUDAInverse.cu
BuildBx.cu

Each one should be built using the following command on MATLAB
system('nvcc -ptx CUDAMul.cu')
system('nvcc -ptx CUDAInverse.cu')
system('nvcc -ptx BuildBx.cu')

After the execution of all the 3 commands, user should have following 3 files in the project folder.
CUDAMul.ptx
CUDAInverse.ptx
BuildBx.ptx

Runnung procedure:

1. Enter “RunFACTORS” in MATLAB work area OR open the file "RunFACTORS.m" and run the file from editor. A pop-up window showing all the Excel spreadsheet data files mentioned above will appear on your screen.
2. User has to double click on one of the files and the program will then execute the program on that file. 
3. If you want to edit any data file, just use Excel spreadsheet program, and be sure to SAVE the file before running the MATLAB program.
4. Program displays PTDF and LODF data’s for the combinations as per the connection.

Example Test INPUT:

Spreadsheet Used:6BusCase_networkdata.xlsx

Test OUTPUT:

Case ID: 6BusCase_networkdata.xlsx 

radial_bus_location =

   Empty matrix: 0-by-1

AFACT MATRIX
Monitored      GENERATOR
Line           

                  1           2           3           4           5           6       

 1 to  2       0.0000     -3.3130     -2.8339     -2.2167     -2.2649     -2.8611    
 1 to  4       0.0000     -2.7100     -2.3436     -2.1069     -1.9084     -2.3644    
 1 to  5       0.0000     -3.8118     -3.3796     -2.5876     -2.8664     -3.4042    
 2 to  3       0.0000     -0.8793     -1.1403     -0.6088     -0.7441     -0.9971    
 2 to  4       0.0000      1.2061      0.9807      0.2196      0.7129      0.9935    
 2 to  5       0.0000     -1.6031     -1.4903     -1.1098     -1.3565     -1.4967    
 2 to  6       0.0000     -1.0368     -1.1839     -0.7178     -0.8773     -1.3608    
 3 to  5       0.0000     -1.0042     -0.6232     -0.6952     -0.8497     -0.7683    
 3 to  6       0.0000      0.1248      0.4829      0.0864      0.1056     -0.2288    
 4 to  5       0.0000     -1.5038     -1.3629     -0.8873     -1.1956     -1.3709    
 5 to  6       0.0000      0.9119      0.7011      0.6313      0.7716      0.5896    


POWER TRANSFER DISTRIBUTION FACTOR (PTDF) MATRIX
Monitored      Transaction
Line           From(Sell) - To(Buy)

               1 to  2     1 to  4     1 to  5     2 to  3     2 to  4     2 to  5     2 to  6     3 to  5     3 to  6     4 to  5     5 to  6    

 1 to  2       0.4706      0.3149      0.3217     -0.0681     -0.1557     -0.1489     -0.0642     -0.0808      0.0039      0.0068      0.0847    
 1 to  4       0.3149      0.5044      0.2711     -0.0200      0.1895     -0.0438     -0.0189     -0.0238      0.0011     -0.2333      0.0249    
 1 to  5       0.2145      0.1807      0.4072      0.0881     -0.0338      0.1927      0.0831      0.1046     -0.0050      0.2264     -0.1096    
 2 to  3      -0.0544     -0.0160      0.1057      0.3960      0.0384      0.1601      0.2451     -0.2359     -0.1509      0.1217      0.0850    
 2 to  4      -0.3115      0.3790     -0.1013      0.0961      0.6904      0.2102      0.0906      0.1141     -0.0055     -0.4802     -0.1196    
 2 to  5      -0.0993     -0.0292      0.1927      0.1335      0.0701      0.2919      0.1259      0.1585     -0.0076      0.2219     -0.1661    
 2 to  6      -0.0642     -0.0189      0.1246      0.3064      0.0453      0.1888      0.4742     -0.1176      0.1678      0.1435      0.2854    
 3 to  5      -0.0622     -0.0183      0.1207     -0.2268      0.0439      0.1829     -0.0905      0.4097      0.1363      0.1390     -0.2733    
 3 to  6       0.0077      0.0023     -0.0150     -0.3772     -0.0055     -0.0227      0.3356      0.3545      0.7128     -0.0173      0.3583    
 4 to  5       0.0034     -0.1166      0.1698      0.0761     -0.1201      0.1664      0.0717      0.0903     -0.0043      0.2865     -0.0947    
 5 to  6       0.0565      0.0166     -0.1096      0.0708     -0.0399     -0.1661      0.1902     -0.2369      0.1194     -0.1262      0.3563    


LINE OUTAGE DISTRIBUTION FACTOR (LODF) MATRIX
Monitored      Outage of one circuit
Line           From - To

               1 to  2     1 to  4     1 to  5     2 to  3     2 to  4     2 to  5     2 to  6     3 to  5     3 to  6     4 to  5     5 to  6    

 1 to  2       0.0000      0.6353      0.5427     -0.1127     -0.5031     -0.2103     -0.1221     -0.1369      0.0135      0.0096      0.1316    
 1 to  4       0.5948      0.0000      0.4573     -0.0331      0.6121     -0.0618     -0.0359     -0.0403      0.0040     -0.3269      0.0387    
 1 to  5       0.4052      0.3647      0.0000      0.1458     -0.1090      0.2721      0.1580      0.1772     -0.0174      0.3174     -0.1703    
 2 to  3      -0.1029     -0.0323      0.1783      0.0000      0.1242      0.2262      0.4662     -0.3995     -0.5253      0.1706      0.1320    
 2 to  4      -0.5884      0.7647     -0.1708      0.1591      0.0000      0.2969      0.1724      0.1933     -0.0190     -0.6731     -0.1858    
 2 to  5      -0.1875     -0.0589      0.3250      0.2209      0.2264      0.0000      0.2394      0.2685     -0.0264      0.3110     -0.2580    
 2 to  6      -0.1213     -0.0381      0.2102      0.5073      0.1464      0.2667      0.0000     -0.1992      0.5842      0.2011      0.4433    
 3 to  5      -0.1175     -0.0369      0.2036     -0.3755      0.1418      0.2583     -0.1720      0.0000      0.4747      0.1948     -0.4246    
 3 to  6       0.0146      0.0046     -0.0253     -0.6245     -0.0176     -0.0321      0.6382      0.6005      0.0000     -0.0242      0.5567    
 4 to  5       0.0065     -0.2353      0.2865      0.1259     -0.3879      0.2350      0.1365      0.1530     -0.0150      0.0000     -0.1471    
 5 to  6       0.1067      0.0335     -0.1849      0.1172     -0.1288     -0.2346      0.3618     -0.4013      0.4158     -0.1769      0.0000    


>> 

 POWER TRANSFER DISTRIBUTION FACTOR (PTDF) MATRIX displays the impact of transaction on every connection possible. For example 1 to 2 row displays the impact of a transaction on this line on every other connection in the network.

Similarly,
LINE OUTAGE DISTRIBUTION FACTOR (LODF) MATRIX displays the impact of outage on every connection possible.

