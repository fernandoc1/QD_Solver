//---------------------------------------------------------------------------
//      Runge-Kutta to solve Diff. Equation
//      Dr. Jose M. Villas-Boas
//      Universidade Federal de Uberlandia, 
//      Last change, 10/09/2010
//---------------------------------------------------------------------------
#include <cmath>                    // some useful mathematic functions
#include <iostream>                 // Provides C++ input and output fundamentals
#include <fstream>                  // Provides facilities for file-based input and output
#include <iomanip>                  // Provides facilities to manipulate output formatting
#include <complex>
#include <string>
#include <time.h>
#include <omp.h>                    // opemmp for parallel computation
#include <sys/stat.h>               // to use mkdir()
#include <unistd.h>                 // getcwd(), chdir() definitions
#include "qd_solver.h"              // Simple Vector, Matrix class e functions for QD solver
#include <cstdlib>
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------

void printExecutionTime(clock_t* start, clock_t* stop)
{
	printf("Execution time: %6.3fs.\n",((double)*stop-*start)/CLOCKS_PER_SEC);
	return;
}

int main(int argc, char** argv)
{
	clock_t start=clock();
	//nn=10; nl=115; ne=102; nh=nn*nl;              // Global variables
	nn = 10;
	nl = 115;
	ne = 102;
	nh = nn * nl; // Global variables
	/* Command Line Parameters
	 if(argc!=4)
	 {
	 printf("nn, nl, ne\n");
	 return 1;
	 }
	 nn=atoi(argv[1]); nl=atoi(argv[2]); ne=atoi(argv[3]); nh=nn*nl;              // Global variables
	 */
	string dir_name = "dat";
	mkdir(dir_name.c_str(), 0755);
	chdir(dir_name.c_str());

	cout.precision(4);
	cout.setf(std::ios::fixed);

	//            InP    InAlGaAs InGaAs InAlGaAs  InAs    InAs    InP   InAlGaAs  InP
	double me[] =
	{ 0.073, 0.053, 0.043, 0.053, 0.027, 0.027, 0.073, 0.053, 0.073 }; // Effective mass
	double Ve[] =
	{ 0.512, 0.374, 0.248, 0.374, 0.000, 0.000, 0.512, 0.374, 0.512 }; // Effective Potential
	double zs[] =
	{ 2000.0, 990.0, 100.0, 30.0, 20.0, 50.0, 110.0, 990.0, 2000.0 }; // sizes in z direction
	double rs[] =
	{ 1200.0, 1200.0, 1200.0, 1200.0, 1200.0, 250.0, 1200.0, 1200.0, 1200.0 }; // sizes ir r direction

	int ni = sizeof me / sizeof me[0];
	// volt=(0.1)/3000.0e-9*1.0e-5
	//cout<<(0.1)/3000.0e-9*1.0e-5<<endl;
	//---------------------------------------------------------------------------
	double L = 0.0;
	for (int i = 0; i < ni; i++)
	{
		if (rs[i] == rs[0])
			L += zs[i];
	} // L = total size without dot

	int m_ang0 = 0, m_ang1 = 1;
	Matrix<double> Vec0(nh, ne), Vec1(nh, ne);
	Vector<double> En0(nh), En1(nh);

	//omp_set_nested(1);  // set to parallelize nested pragma omp commands
	//#pragma omp parallel sections // starts a new team
	{
		//#pragma omp section         // solve the 2 angular momentum simultaneously.
		{
			qd_solver(m_ang0, me, Ve, zs, rs, En0, Vec0, -2.6);
			wave_funciton(m_ang0, zs, rs, Vec0);
			Oscilator_z(m_ang0, me[5], L, En0, Vec0);
		}
		//#pragma omp section
		{
			qd_solver(m_ang1, me, Ve, zs, rs, En1, Vec1, -2.6);
			wave_funciton(m_ang1, zs, rs, Vec1);
			Oscilator_z(m_ang1, me[5], L, En1, Vec1);
		}
	}

	Oscilator_r(me[5], rs[0], m_ang0, m_ang1, En0, En1, Vec0, Vec1);

	// nl=1000; // Quantum Well solver
	// Matrix<double> Vec(nl,ne);
	// Vector<double> En(nl);
	// qw_solver(me,Ve,zs,En,Vec,-3.0);
	// Oscilator_z_qw(me[5],L,En,Vec);

	clock_t stop=clock();
	//double cpu_time = (t_cpu_stop-t_cpu_start)*1.0/CLOCKS_PER_SEC;
	//cout << "CPU time = "<< cpu_time << " in seconds" << endl;

	printExecutionTime(&start, &stop);

	return 0;
}
//---------------------------------------------------------------------------

