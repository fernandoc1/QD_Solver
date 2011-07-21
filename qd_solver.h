#include <cmath>                    // some useful mathematic functions
#include <iostream>                 // Provides C++ input and output fundamentals
#include <fstream>                  // Provides facilities for file-based input and output
#include <iomanip>                  // Provides facilities to manipulate output formatting
#include <complex>
#include <string>
#include <cstdlib>

using namespace std;

typedef std::complex<double> dcomplex;

//---------------------------------------------------------------------------
//          Simple Vector Class
//          Works well for small vectors and has no bound check! Fast!!!
//          Use carefully
//---------------------------------------------------------------------------
template<class T>
class Vector
{
	int N; // size of the vector
	T *vec; // pointer to data
public:
	// Constructors
	Vector(); // default constructor is called when no arguments are given
	Vector(int N, const T & def_val = 0); // constructor takes size as an argument
	Vector(const Vector & A); // copy constructor necessary for return by value

	// Destructor is required because the constructors above use "new"
	~Vector()
	{
		delete[] vec;
	}

	// Accessors
	T & operator[](int i)
	{
		return vec[i];
	}
	T operator[](int i) const
	{
		return vec[i];
	}

	T & operator()(int i)
	{
		return vec[i];
	}
	T operator()(int i) const
	{
		return vec[i];
	}

	int size() const
	{
		return N;
	} // returns size
};
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
// Default constructor creates empty vector
template<class T>
Vector<T>::Vector() :
		N(0), vec(0)
{
}
;

// Vector is initialization with initial size and default value
template<class T>
Vector<T>::Vector(int N_, const T& def_val) :
		N(N_)
{
	vec = new T[N];
	for (int i = 0; i < N; i++)
		vec[i] = def_val;
}

// Copy constructor needed for return by value
template<class T>
Vector<T>::Vector(const Vector& v) :
		N(v.N)
{
	vec = new T[N];
	for (int i = 0; i < N; i++)
		vec[i] = v.vec[i];
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
// Printing of a vector in Python style.
template<class T>
std::ostream& operator<<(std::ostream& stream, const Vector<T>& vec)
{
	// stream<<"[";
	for (int i = 0; i < vec.size(); i++)
	{
		stream << vec[i];
		if (i != vec.size() - 1)
			stream << std::setw(12);
		// if (i!=vec.size()-1) stream<<", ";
		// else stream<<"]";
	}
	return stream;
}
//---------------------------------------------------------------------------

//    Function which is non member of the vector class

//---------------------------------------------------------------------------
template<class U>
Vector<U> flip(const Vector<U> & v)
{
	int n = v.size();
	Vector<U> res(n);
	for (int i = 0; i < n; i++)
		res[i] = v[n - 1 - i];
	return res;
}

template<class U>
U sum(const Vector<U> & v)
{
	int n = v.size();
	U s = 0;
	for (int i = 0; i < n; i++)
		s += v[i];
	return s;
}

template<class U>
U max(const Vector<U> & v)
{
	int n = v.size();
	U m = v[0];
	for (int i = 0; i < n; i++)
		m = (m > v[i]) ? m : v[i];
	return m;
}

template<class U>
U min(const Vector<U> & v)
{
	int n = v.size();
	U m = v[0];
	for (int i = 0; i < n; i++)
		m = (m < v[i]) ? m : v[i];
	return m;
}

template<class U>
Vector<U> linspace(const U & xi, const U & xf, const int & n)
{
	Vector<U> res(n);
	U dx = (xf - xi) / (n - 1), x = xi;
	for (int i = 0; i < n; i++)
	{
		res[i] = x;
		x += dx;
	}
	return res;
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
template<class U>
Vector<U> conj(const Vector<U> & v)
{
	int n = v.size();
	Vector<U> res(n);
	for (int i = 0; i < n; i++)
		res[i] = conj(v[i]);
	return res;
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
template<class U>
Vector<double> real(const Vector<U> & v)
{
	int n = v.size();
	Vector<double> res(n);
	for (int i = 0; i < n; i++)
		res[i] = real(v[i]);
	return res;
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
template<class U>
Vector<U> imag(const Vector<U> & v)
{
	int n = v.size();
	Vector<U> res(n);
	for (int i = 0; i < n; i++)
		res[i] = imag(v[i]);
	return res;
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
template<class U>
double norm(const U & v)
{
	int n = v.size();
	double res = 0.0;
	// for (int i=0; i<n; i++) res = res + v[i]*v[i];
	for (int i = 0; i < n; i++)
		res = res + pow(std::abs(v[i]), 2);
	return sqrt(res);
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
//          just a shotcut to print
template<class T>
void print(T & pr)
{
	std::cout << pr << std::endl;
}
//---------------------------------------------------------------------------

//          Simple Matrix Class
//          Fast and with bound cheking!
//          Use carefully!

//---------------------------------------------------------------------------
template<class T>
class Matrix
{
	int N, M; // size of the vector
	T *mat; // pointer to data
public:

	// Constructors
	Matrix(); // default constructor is called when no arguments are given
	Matrix(int N, int M, const T & def_val = 0); // constructor takes size as an argument
	Matrix(const Matrix & A); // copy constructor necessary for return by value

	// Destructor is required because the constructors above use "new"
	~Matrix()
	{
		delete[] mat;
	}

	// Accessors

	// fortran like access: Fortran is column major, so that the leftmost subscript varies the fastest in memory.
	T & operator()(int i, int j)
	{
		return mat[N * j + i];
	}
	T operator()(int i, int j) const
	{
		return mat[N * j + i];
	}

	// T & operator()(int i, int j){ return mat[M*i+j]; }
	// T operator()(int i, int j) const { return mat[M*i+j]; }

	int rows() const
	{
		return N;
	} // returns size
	int cols() const
	{
		return M;
	} // returns size
	void resize(int N_, int M_); // resizes existing vector

	T col_col(const int j1, const Matrix<T>& rhs, const int j2);
};
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
// Default constructor creates empty vector
template<class T>
Matrix<T>::Matrix() :
		N(0), M(0), mat(0)
{
}
;

// Matrix is initialization with initial size and default value
template<class T>
Matrix<T>::Matrix(int N_, int M_, const T& def_val) :
		N(N_), M(M_)
{
	mat = new T[N * M];for (int i=0; i<N*M; i++) mat[i] = def_val;
}

// Copy constructor needed for return by value
template<class T>
Matrix<T>::Matrix(const Matrix& ma) :
		N(ma.N), M(ma.M)
{
	mat = new T[N * M];for (int i=0; i<N*M; i++) mat[i] = ma.mat[i];
}
//---------------------------------------------------------------------------

// Matrix functions which is not member of Matrix class

//---------------------------------------------------------------------------
// Prints matrix in formated style
template<class T>
std::ostream& operator<<(std::ostream & stream, const Matrix<T> & ma)
{
	int n = ma.rows(), m = ma.cols();
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			stream << std::setw(12) << ma(i, j);
		}
		stream << std::endl;
	}
	return stream; // We have to return stream!
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
// write a matrix in formated style to a file;
template<class T, class V>
void write_array(V fn, const Matrix<T> & ma, int prec = 8)
{
	std::ofstream file(fn, std::ios::out);
	file.precision(prec);
	file.setf(std::ios::fixed | std::ios::showpoint);
	int n = ma.rows(), m = ma.cols();
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			file << std::setw(12) << ma(i, j);
		}
		file << std::endl;
	}
	file.close();
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
//           convert integer to string
std::string itoa(int num)
{
	std::stringstream converter;
	converter << num;
	return converter.str();
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
// if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j;   fortran
// if UPLO = 'L', AP(i + (j-1)*(2*n-j)/2) = A(i,j) for j<=i<=n.   fortran

// if UPLO = 'U', AP(i + (j+1)*j/2) = A(i,j) for 1<=i<=j;   c++
// if UPLO = 'L', AP(i + j*(2*n-j-1)/2) = A(i,j) for j<=i<=n.   c++

// Example
// for( i = 0; i < N; i++ ) {
//     for( j = 0; j < N; j++ ) {
//         if(i<=j) {  //  UPLO = 'U'   ou (j>=i)
//             k=i+(j+1)*j/2;
//             Hp[k]=H(i,j);
//         }
//         if(i>=j) {   //  UPLO = 'L'  ou (j<=i)
//              k=i+j*(2*N-j-1)/2;
//              Hp[k]=H(i,j);
//         }
//     }
// }
extern "C" void dspevx_(char *jobz, char *range, char *uplo, int *n, double *ap,
		double *vl, double *vu, int *il, int *iu, double *abstol, int *m,
		double *w, double *z, int *ldz, double *work, int *iwork, int *ifail,
		int *info);

// extern "C" void dspevx_(char jobz, char range, char uplo, int n, double *da,
//                 double dvl, double dvu, int il, int iu, double dabtol,
//                 int *nfound, double *dw, double *dz, int ldz, int
//                 *ifail, int *info);
// extern "C" void sspevx_(char jobz, char range, char uplo, int n, float *sa,
//                 float svl, float svu, int il, int iu, float sabtol, int
//                 *nfound, float *sw, float *sz, int ldz, int *ifail, int
//                 *info);

void Eig_p(Vector<double> &Hp, Vector<double> &En, Matrix<double> &Vec,
		char uplo = 'U', int ne = 10)
{
	int n = En.size();
	double work[8 * n];int
	iwork[5 * n], ifail[n];
	char jobz = 'V', range = 'I';
	double abstol = -1, vl, vu;
	int il = 1, iu = ne, m, lda = n, ldz = n, info, lwork;

	dspevx_(&jobz, &range, &uplo, &n, &Hp[0], &vl, &vu, &il, &iu, &abstol, &m,
			&En[0], &Vec(0, 0), &ldz, work, iwork, ifail, &info);
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
//                  Eigenvalue and optinaly the eigenvectors of a real symetric Matrix
extern "C" void dsyev_(const char* jobz, const char* uplo, const int* n,
		double* A, const int* lda, double* w, double* work, const int* lwork,
		int* info);

// Notice that fortran is column major, so that the leftmost subscript varies the fastest in memory.
// So 'U' in this Matrix class is 'L' in fortran array.
inline Vector<double> Eig(Matrix<double>& H, Vector<double>& En,
		char jobz = 'N', char uplo = 'U')
{
	if (H.rows() != H.cols())
	{
		std::cerr << "Can not diagonalize non-square matrix!" << std::endl;
	}
	int N = H.rows(), lwork = 4 * N, info = 0;
	static Vector<double> work(lwork);
	dsyev_(&jobz, &uplo, &N, &H(0, 0), &N, &En[0], &work[0], &lwork, &info);
	return En;
}

inline Vector<double> Eig(Matrix<double>& H, Vector<double>& En,
		Matrix<double>& Vec, char uplo = 'U')
{
	if (H.rows() != H.cols())
	{
		std::cerr << "Can not diagonalize non-square matrix!" << std::endl;
	}
	int N = H.rows(), lwork = 4 * N, info = 0;
	static Vector<double> work(lwork);
	// dsyev_("V", "U", &N, &H(0,0), &N, &En[0], &work[0], &lwork, &info);
	dsyev_("V", &uplo, &N, &H(0, 0), &N, &En[0], &work[0], &lwork, &info);
	// Vec=trans(H);
	return En;
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
double qgaus(double func(const double), const double a, const double b)
{
	static const double x[] =
	{ 0.1488743389816312, 0.4333953941292472, 0.6794095682990244,
			0.8650633666889845, 0.9739065285171717 };
	static const double w[] =
	{ 0.2955242247147529, 0.2692667193099963, 0.2190863625159821,
			0.1494513491505806, 0.0666713443086881 };
	int j;
	double xr, xm, dx, s;
	xm = 0.5 * (b + a);
	xr = 0.5 * (b - a);
	s = 0;
	for (j = 0; j < 5; j++)
	{
		dx = xr * x[j];
		s += w[j] * (func(xm + dx) + func(xm - dx));
	}
	return s *= xr;
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
void gauleg(const double x1, const double x2, Vector<double> &x,
		Vector<double> &w)
{
	const double EPS = 1.0e-14;
	int m, j, i;
	double z1, z, xm, xl, pp, p3, p2, p1;
	int n = x.size();
	m = (n + 1) / 2;
	xm = 0.5 * (x2 + x1);
	xl = 0.5 * (x2 - x1);
	for (i = 0; i < m; i++)
	{
		z = cos(3.141592654 * (i + 0.75) / (n + 0.5));
		do
		{
			p1 = 1.0;
			p2 = 0.0;
			for (j = 0; j < n; j++)
			{
				p3 = p2;
				p2 = p1;
				p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1);
			}
			pp = n * (z * p1 - p2) / (z * z - 1.0);
			z1 = z;
			z = z1 - p1 / pp;
		} while (fabs(z - z1) > EPS);
		x[i] = xm - xl * z;
		x[n - 1 - i] = xm + xl * z;
		w[i] = 2.0 * xl / ((1.0 - z * z) * pp * pp);
		w[n - 1 - i] = w[i];
	}
}

const int nmax = 120;
Matrix<double> xtab(nmax, nmax), wtab(nmax, nmax);

void gauleg_table()
{
	double xi = 0.0, xf = 1.0;
	for (size_t i = 1; i < nmax; i++)
	{
		Vector<double> w(i), x(i);
		gauleg(xi, xf, x, w);
		for (size_t j = 0; j < nmax; j++)
		{
			xtab(i, j) = x(j);
			wtab(i, j) = w(j);
		}
	}
}

double gauleg_i(double func(const double), const double x1, const double x2,
		const int n)
{
	double s = 0.0;
	double xr = (x2 - x1);
	for (int i = 0; i < n; i++)
	{
		s += (xr * wtab(n, i) * func(xr * xtab(n, i)));
	}
	return s;
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
void ZerroBesselJN(int m, int nz, Vector<double> & v)
{
	double x, dx, bs0, erro;
	erro = 1e-10;
	x = 0.0;
	for (int iz = 1; iz < nz; ++iz)
	{
		dx = 0.01;
		x = x + dx;
		bs0 = jn(m, x);
		while (2 * dx > erro)
		{
			x = x + dx;
			if (bs0 * jn(m, x) < 0.0)
			{
				x = x - dx;
				dx = dx / 2;
			}
		}
		v[iz] = x;
	}
}
//---------------------------------------------------------------------------
//   Functions to obtain the Eigenstate of a QD using cylindrical symmetry. //
//                        Based on fortran version 8                        //
//---------------------------------------------------------------------------
using namespace std;
typedef complex<double> my_complex;
const double h2m = 3.809984039;
const double Pi = M_PI;
const my_complex I(0.0, 1.0);
Matrix<double> zJn(100, 1000); // zJn(m_ang_max,n_max)

int nn, nl, nh, ne;
//---------------------------------------------------------------------------
double fz(int n, double z)
{
	return sqrt(2.0) * sin(n * Pi * z);
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
double Zm(int n, int m)
{
	double fn_res;
	if (n == m)
	{
		fn_res = 0.5;
	}
	else
	{
		fn_res = 4.0 * (pow(-1.0, m + n) - 1) * m * n
				/ (pow(pow(m, 2.) - pow(n, 2.), 2.) * pow(Pi, 2.));
	}
	return fn_res;
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
double Zint(int n, int m, double a, double b)
{
	double fn_res;
	if (n == m)
	{
		fn_res = (b - sin(2 * n * Pi * b) / (2.0 * n * Pi))
				- (a - sin(2 * n * Pi * a) / (2.0 * n * Pi));
	}
	else
	{
		fn_res = 2.0 / Pi
				* (n * cos(n * Pi * b) * sin(m * Pi * b)
						- m * cos(m * Pi * b) * sin(n * Pi * b))
				/ (pow(m, 2.) - pow(n, 2.))
				- 2.0 / Pi
						* (n * cos(n * Pi * a) * sin(m * Pi * a)
								- m * cos(m * Pi * a) * sin(n * Pi * a))
						/ (pow(m, 2.) - pow(n, 2.));
	}
	return fn_res;
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
double dZint(int n, int m, double a, double b)
{
	double fn_res;
	if (n == m)
	{
		fn_res = (n * Pi * (2 * n * Pi * b + sin(2 * n * Pi * b))) / 2.0
				- (n * Pi * (2 * n * Pi * a + sin(2 * n * Pi * a))) / 2.0;
	}
	else
	{
		fn_res = m * n * Pi
				* (sin((m - n) * Pi * b) / (m - n)
						+ sin((m + n) * Pi * b) / (m + n))
				- m * n * Pi
						* (sin((m - n) * Pi * a) / (m - n)
								+ sin((m + n) * Pi * a) / (m + n));
	}
	return fn_res;
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
void ZerroBessel_jn(int nm, int nz)
{
	double x, dx, bs0, erro;
	// Matrix<double> zJn (nm,nz+1,0.0);
	erro = 1e-10;
	for (int im = 0; im <= nm; im++)
	{
		x = 0.0;
		for (int iz = 1; iz < nz; iz++)
		{
			dx = 0.01;
			x = x + dx;
			bs0 = jn(im, x);
			while (2 * dx > erro)
			{
				x = x + dx;
				if (bs0 * jn(im, x) < 0.0)
				{
					x = x - dx;
					dx = dx / 2;
				}
			}
			zJn(im, iz) = x;
		}
	}
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
double inline Rnm(int n, int m, double r)
{
	return sqrt(2.0) / (fabs(jn(m + 1, zJn(m, n)))) * jn(m, zJn(m, n) * r);
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
double dRnm(int n, int m, double r)
{
	return sqrt(2.0) / (abs(jn(m + 1, zJn(m, n))))
			* (jn(m - 1, zJn(m, n) * r) - jn(m + 1, zJn(m, n) * r)) / 2
			* (zJn(m, n));
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
double dot_shape(double h0, double ro, double r)
{
	// return h0*(1.0-r/ro);                // cone
	// return h0*(1.0-(r/ro)**2);           // Lens-shaped Wrong!!!!
	return h0 * sqrt(1.0 - pow(r / ro, 2.)); // Lens-shaped
	// return ro;                             // Cylinder-shaped
	// return h0*exp(-2*r**2/ro**2);          // Gaussian
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
double delta(int n, int m)
{
	return n == m ? 1.0 : 0.0;
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
double f1(double r, int n1, int n2, int l1, int l2, int m_ang, double zdot,
		double h0, double ro)
{
	double z1, z2;
	z1 = zdot;
	z2 = zdot + dot_shape(h0, ro, r);
	return r * Rnm(n1, m_ang, r) * Rnm(n2, m_ang, r) * Zint(l1, l2, z1, z2);
}

double RInt(int n1, int n2, int l1, int l2, int m_ang, double zdot, double h0,
		double ro)
{
	int n = (n1 > n2) ? n1 + 8 : n2 + 8;
	double s = 0.0;
	double xr = (ro - 0.0);
	for (int i = 0; i < n; i++)
	{
		s += xr * wtab(n, i)
				* f1(xr * xtab(n, i), n1, n2, l1, l2, m_ang, zdot, h0, ro);
	}
	return s;
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
double f2(double r, int n1, int n2, int l1, int l2, int m_ang, double zdot,
		double h0, double ro, double r0, double L)
{
	double z1, z2;
	z1 = zdot;
	z2 = zdot + dot_shape(h0, ro, r);
	return r / pow(r0, 2.) * dRnm(n1, m_ang, r) * dRnm(n2, m_ang, r)
			* Zint(l1, l2, z1, z2)
			+ pow(m_ang, 2.) / r * Rnm(n1, m_ang, r) * Rnm(n2, m_ang, r)
					* Zint(l1, l2, z1, z2) / pow(r0, 2.)
			+ r / pow(L, 2.) * Rnm(n1, m_ang, r) * Rnm(n2, m_ang, r)
					* dZint(l1, l2, z1, z2);
}

double dRInt(int n1, int n2, int l1, int l2, int m_ang, double zdot, double h0,
		double ro, double r0, double L)
{
	int n = (n1 > n2) ? n1 + 8 : n2 + 8;
	double s = 0.0;
	double xr = (ro - 0.0);
	for (int i = 0; i < n; i++)
	{
		s += xr * wtab(n, i)
				* f2(xr * xtab(n, i), n1, n2, l1, l2, m_ang, zdot, h0, ro, r0,
						L);
	}
	return s;
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
double f3(double r, int n1, int n2, int m1, int m2)
{
	return r * r / 2 * Rnm(n1, m1, r) * Rnm(n2, m2, r);
}

double rmed(int n1, int n2, int m1, int m2)
{
	int n = (n1 > n2) ? n1 + 8 : n2 + 8;
	double s = 0.0;
	double xr = (1.0 - 0.0);
	for (int i = 0; i < n; i++)
	{
		s += xr * wtab(n, i) * f3(xr * xtab(n, i), n1, n2, m1, m2);
	}
	return s;
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
template<class T, class V>
void qd_solver(int m_ang, T &me, T &Ve, T &zsize, T &rsize, Vector<double> &E,
		V &Vec, double eF = 0.0)
{
	Vector<double> Hp(nh * (nh + 1) / 2); // Matrix H in package format to use with Lapack dspevx
	size_t ni = sizeof me / sizeof me[0]; // ni=number of interfaces

	double eFd = eF * 1.0e-5; // eF in unity of [KV/cm], converting to [V/A]

	gauleg_table(); // generate a table for the gauss legendre quadrature integral
	ZerroBessel_jn(m_ang + 2, nn + 10); // generate a table of Zero of the Bessel Functions

	double h0, r0, L = 0.0;
	for (size_t i = 0; i < ni; ++i)
	{
		if (rsize[i] == rsize[0])
			L += zsize[i];
	} // L = total size without dot

	double z_shift = -0.5; // make the external files symetric on z

	Vector<double> rs(ni), zs(ni), zpos(ni + 1);
	ofstream shape("dots.dat", ios::out);
	ofstream pot("Potential.dat", ios::out);
	r0 = rsize[0];
	zpos[0] = 0.0;
	int nr = 501;
	for (int k = 0; k < ni; k++)
	{
		double d_zs; // size of the dot, used to build the potential on z at dot position
		zs[k] = zsize[k] / L; // normalized zsize
		rs[k] = rsize[k] / r0; // normalized rsize
		if (rs[k] != rs[0])
		{ // this is a dot!
			zpos[k + 1] = zpos[k]; // if it is a QD, the next interface position start at the WL
			double dr = (rs[k] - 0.0) / (nr - 1); // writing the QD shape in a file
			for (int i = 0; i < nr; i++)
			{
				shape
						<< i * dr * r0 / 10
						<< " "
						<< (z_shift + zpos[k] + dot_shape(zs[k], rs[k], i * dr))
								* L / 10 << endl;
			}
			d_zs = zs[k]; // if it is a QD, used to build the potential on z at dot position
			pot << (z_shift + zpos[k]) * L / 10 << " "
					<< Ve[k] - eF * (z_shift + zpos[k]) * L * 1.0e-5 << endl;
			pot << (z_shift + zpos[k + 1] + d_zs) * L / 10 << " "
					<< Ve[k] - eF * (z_shift + zpos[k + 1] + d_zs) * L * 1.0e-5
					<< endl;
		}
		else
		{
			zpos[k + 1] = zpos[k] + zs[k];
			pot << (z_shift + zpos[k] + d_zs) * L / 10 << " "
					<< Ve[k] - eF * (z_shift + zpos[k] + d_zs) * L * 1.0e-5
					<< endl;
			pot << (z_shift + zpos[k + 1]) * L / 10 << " "
					<< Ve[k] - eF * (z_shift + zpos[k + 1]) * L * 1.0e-5
					<< endl;
			d_zs = 0.0;
		}
	}
	pot.close();
	shape.close();

	ofstream wells("wells.dat", ios::out);
	wells << 0.0;
	for (int k = 1; k < ni; k++)
	{
		if (rs[k] == rs[0])
			wells << " " << (z_shift + zpos[k]) * L / 10;
	}
	wells << endl;
	wells << r0 / 10;
	for (int k = 1; k < ni; k++)
	{
		if (rs[k] == rs[0])
			wells << " " << (z_shift + zpos[k]) * L / 10;
	}
	wells.close();

	int i = 0, j = 0;
	// doing a parallel loop to speed up in multiprocessor computers
	//#pragma omp parallel for schedule(dynamic, 2) private(i,j) shared(Hp)
	for (int n1 = 1; n1 <= nn; n1++)
	{ // notice the index starting from 1;
		for (int l1 = 1; l1 <= nl; l1++)
		{ // notice the index starting from 1;
			i = (n1 - 1) * nl + (l1 - 1);
			for (int n2 = 1; n2 <= nn; n2++)
			{
				for (int l2 = 1; l2 <= nl; l2++)
				{
					j = (n2 - 1) * nl + (l2 - 1);
					if (j >= i)
					{
						double kin_well = 0.0, pot_well = 0.0, kin_dot = 0.0,
								pot_dot = 0.0;
						for (int k = 0; k < ni; k++)
						{
							if (rs[k] != rs[0])
							{ // This is a dot?
								kin_dot += h2m * (1.0 / me[k] - 1.0 / me[k + 1])
										* dRInt(n1, n2, l1, l2, m_ang, zpos[k],
												zs[k], rs[k], r0, L);
								pot_dot += (Ve[k] - Ve[k + 1])
										* RInt(n1, n2, l1, l2, m_ang, zpos[k],
												zs[k], rs[k]);
							}
							else
							{ // This is a Quantum Well or Barrier
								kin_well += h2m / me[k]
										* dZint(l1, l2, zpos[k],
												zpos[k] + zs[k]) / pow(L, 2.)
										* delta(n1, n2);
								pot_well += (Ve[k]
										+ h2m / me[k]
												* pow(zJn(m_ang, n1) / r0, 2.))
										* Zint(l1, l2, zpos[k], zpos[k] + zs[k])
										* delta(n1, n2);
							}
						}
						Hp[i + (j + 1) * j / 2] = kin_well + pot_well + kin_dot
								+ pot_dot
								- eFd * L * Zm(l1, l2) * delta(n1, n2);
					}
				}
			}
		}
	}
	clock_t t_cpu_start = clock(); // to compute cpu_time
	Eig_p(Hp, E, Vec, 'U', ne);
	clock_t t_cpu_stop = clock(); // to compute cpu_time
	//double cpu_time = (t_cpu_stop-t_cpu_start)*1.0/CLOCKS_PER_SEC;
	//cout << "Eig time = "<< cpu_time << " in seconds" << endl;

	string fname = "Energy_m" + itoa(m_ang) + ".dat";
	ofstream file(fname.c_str(), ios::out);
	file.precision(4);
	// file.setf(ios::fixed,ios::floatfield);

	for (int i = 0; i < ne; ++i)
	{
		cout << E[i] << " ";
	}
	cout << endl;
	file << (z_shift + 0.0) * L / 10;
	for (int i = 0; i < ne; ++i)
	{
		file << " " << E[i];
	}
	file << endl;
	file << (z_shift + 1.0) * L / 10;
	for (int i = 0; i < ne; ++i)
	{
		file << " " << E[i];
	}
	file << endl;
	file.close();
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
template<class T, class V>
void wave_funciton(int m_ang, T &zs, T &rs, V &Vec)
{
	int nz = 301, nr = 251;
	size_t ni = sizeof zs / sizeof zs[0];
	// size_t ni=zs.size();
	double r0 = rs[0], L = 0.0;

	for (int i = 0; i < ni; ++i)
	{
		if (rs[i] == rs[0])
			L += zs[i];
	}
	double zi = zs[0] / L, zf = (L - zs[ni - 1]) / L;
	double z_shift = -0.5;

	double dz = (zf - zi) / (nz - 1), dr = 1.0 / (nr - 1);
	double z, r, psy;

	//----------------------
	Matrix<double> rn(nr, nn + 1), zl(nz, nl + 1);
	for (size_t ir = 0; ir < nr; ir++)
	{
		r = ir * dr;
		for (size_t n = 1; n <= nn; n++)
		{
			rn(ir, n) = Rnm(n, m_ang, r);
		}
	}

	for (size_t iz = 0; iz < nz; iz++)
	{
		z = zi + iz * dz;
		for (size_t l = 1; l <= nl; l++)
		{
			zl(iz, l) = fz(l, z);
		}
	}

	for (size_t j = 0; j < ne; ++j)
	{
		string fname = "psy_" + itoa(j + 1) + "_m" + itoa(m_ang) + ".dat";
		ofstream file(fname.c_str(), ios::out);
		file.precision(6);
		// file.setf(ios::fixed,ios::floatfield);
		file << " " << 0.0 / 10 << " " << r0 / 10 << " "
				<< (z_shift + zi) * L / 10 << " " << (z_shift + zf) * L / 10
				<< endl; // x and y scale
		for (size_t iz = 0; iz < nz; iz++)
		{
			for (size_t ir = 0; ir < nr; ir++)
			{
				psy = 0.0;
				for (size_t n = 1; n <= nn; n++)
				{
					for (size_t l = 1; l <= nl; l++)
					{
						int i = (n - 1) * nl + (l - 1);
						psy += Vec(i, j) * zl(iz, l) * rn(ir, n);
					}
				}
				file << " " << psy;
			}
			file << endl;
		}
		file.close();
	}
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
void Oscilator_z(int m_ang, double m_dot, double L, Vector<double> E,
		Matrix<double> &Vec)
{
	double z_av, os;
	int i, j;
	Matrix<double> Zmed(nl + 1, nl + 1);
	for (size_t l1 = 1; l1 <= nl; l1++)
	{
		for (size_t l2 = 1; l2 <= nl; l2++)
		{
			Zmed(l1, l2) = L * Zm(l1, l2);
		}
	}

	string fname = "Osc_strain_Z_m" + itoa(m_ang) + ".dat";
	ofstream file(fname.c_str(), ios::out);
	file.setf(ios::fixed, ios::floatfield);

	for (size_t e1 = 0; e1 < ne; e1++)
	{
		for (size_t e2 = 0; e2 < ne; e2++)
		{
			z_av = 0.0;
			for (size_t n = 1; n <= nn; n++)
			{
				for (size_t l1 = 1; l1 <= nl; l1++)
				{
					i = (n - 1) * nl + (l1 - 1);
					for (size_t l2 = 1; l2 <= nl; l2++)
					{
						j = (n - 1) * nl + (l2 - 1);
						z_av += Vec(i, e1) * Vec(j, e2) * Zmed(l1, l2);
					}
				}
			}
			os = m_dot / h2m * (E[e2] - E[e1]) * pow(z_av, 2.);
			file << "E=" << setw(3) << e1 + 1 << " --> E=" << setw(3) << e2 + 1
					<< ", F=" << setw(8) << setprecision(4) << os << ", ∆E="
					<< setw(7) << setprecision(2) << (E[e2] - E[e1]) * 1000
					<< " meV" << endl;
		}
	}
	file.close();
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
//                       Oscillator Strain for x,y component
template<class T, class V>
void Oscilator_r(double m_dot, double r0, int m1_, int m2_, T E0, T E1, V &Vec0,
		V &Vec1)
{
	double r_av, os;
	Matrix<double> Rm(nn + 1, nn + 1);

	ofstream file("Osc_strain_r.dat", ios::out);
	file.setf(ios::fixed, ios::floatfield);

	for (int im = 0; im < 2; im++)
	{
		int m1 = m1_ + im, m2 = m2_ - im;
		for (size_t n1 = 1; n1 <= nn; n1++)
		{
			for (size_t n2 = 1; n2 <= nn; n2++)
			{
				Rm(n1, n2) = r0 * rmed(n1, n2, m1, m2);
			}
		}
		double De;
		for (size_t e1 = 0; e1 < ne; e1++)
		{
			for (size_t e2 = 0; e2 < ne; e2++)
			{
				r_av = 0.0;
				for (size_t n1 = 1; n1 <= nn; n1++)
				{
					for (size_t n2 = 1; n2 <= nn; n2++)
					{
						for (size_t l = 1; l <= nl; l++)
						{
							int i = (n1 - 1) * nl + (l - 1);
							int j = (n2 - 1) * nl + (l - 1);
							if (m1 == m2 - 1)
							{
								r_av += Vec0(i, e1) * Vec1(j, e2) * Rm(n1, n2);
								De = E1[e2] - E0[e1];
							}
							if (m1 == m2 + 1)
							{
								r_av += Vec1(i, e1) * Vec0(j, e2) * Rm(n1, n2);
								De = E0[e2] - E1[e1];
							}
						}
					}
				}
				os = m_dot / h2m * De * pow(r_av, 2.);
				file << "(" << setw(1) << m1 << "," << setw(3) << e1 + 1
						<< ") --> (" << setw(1) << m2 << "," << setw(3)
						<< e2 + 1 << "), F=" << setw(8) << setprecision(4) << os
						<< ", ∆E=" << setw(7) << setprecision(2) << De * 1000
						<< " meV" << endl;
			}
		}
	}
	file.close();
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
template<class T, class V>
void qw_solver(T &me, T &Ve, T &zsize, Vector<double> &E, V &Vec, double eF =
		0.0)
{
	Vector<double> Hp(nl * (nl + 1) / 2); // Matrix H in package format to use with Lapack dspevx
	size_t ni = sizeof me / sizeof me[0]; // ni=number of interfaces
	double eFd = eF * 1.0e-5; // eF in unity of [KV/cm], converting to [V/A]

	double L = 0.0;
	for (size_t i = 0; i < ni; ++i)
	{
		L += zsize[i];
	} // L = total size without dot

	double z_shift = -0.5; // make the external files symetric on z

	Vector<double> zs(ni), zpos(ni + 1);
	ofstream pot("Potential.dat", ios::out);
	zpos[0] = 0.0;
	for (int k = 0; k < ni; k++)
	{
		zs[k] = zsize[k] / L; // normalized zsize
		zpos[k + 1] = zpos[k] + zs[k];
		pot << (z_shift + zpos[k]) * L / 10 << " "
				<< Ve[k] - eF * (z_shift + zpos[k]) * L * 1.0e-5 << endl;
		pot << (z_shift + zpos[k + 1]) * L / 10 << " "
				<< Ve[k] - eF * (z_shift + zpos[k + 1]) * L * 1.0e-5 << endl;
	}
	pot.close();

	// doing a parallel loop to speed up in multiprocessor computers
#pragma omp parallel for schedule(dynamic, 2) shared(Hp)
	for (int i = 0; i < nl; i++)
	{ // notice the index starting from 1;
		for (int j = 0; j < nl; j++)
		{
			int l1 = i + 1, l2 = j + 1;
			if (j >= i)
			{
				double kin_well = 0.0, pot_well = 0.0;
				for (int k = 0; k < ni; k++)
				{
					kin_well += h2m / me[k]
							* dZint(l1, l2, zpos[k], zpos[k] + zs[k])
							/ pow(L, 2.);
					pot_well += Ve[k] * Zint(l1, l2, zpos[k], zpos[k] + zs[k]);
				}
				Hp[i + (j + 1) * j / 2] = kin_well + pot_well
						- eFd * L * Zm(l1, l2);
			}
		}
	}
	clock_t t_cpu_start = clock(); // to compute cpu_time
	Eig_p(Hp, E, Vec, 'U', ne);
	clock_t t_cpu_stop = clock(); // to compute cpu_time
	//double cpu_time = (t_cpu_stop-t_cpu_start)*1.0/CLOCKS_PER_SEC;
	//cout << "Eig time = "<< cpu_time << " in seconds" << endl;

	string fname = "Energy.dat";
	ofstream file(fname.c_str(), ios::out);
	file.precision(4);
	// file.setf(ios::fixed,ios::floatfield);

	for (int i = 0; i < ne; ++i)
	{
		cout << E[i] << " ";
	}
	cout << endl;
	file << (z_shift + 0.0) * L / 10;
	for (int i = 0; i < ne; ++i)
	{
		file << " " << E[i];
	}
	file << endl;
	file << (z_shift + 1.0) * L / 10;
	for (int i = 0; i < ne; ++i)
	{
		file << " " << E[i];
	}
	file << endl;
	file.close();
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
void Oscilator_z_qw(double m_dot, double L, Vector<double> E,
		Matrix<double> &Vec)
{
	double z_av, os;
	int i, j;
	Matrix<double> Zmed(nl + 1, nl + 1);
	for (size_t l1 = 1; l1 <= nl; l1++)
	{
		for (size_t l2 = 1; l2 <= nl; l2++)
		{
			Zmed(l1, l2) = L * Zm(l1, l2);
		}
	}

	string fname = "Osc_strain_Z.dat";
	ofstream file(fname.c_str(), ios::out);
	file.setf(ios::fixed, ios::floatfield);

	for (int e1 = 0; e1 < ne; e1++)
	{
		for (int e2 = 0; e2 < ne; e2++)
		{
			z_av = 0.0;
			for (int i = 0; i < nl; i++)
			{
				for (int j = 0; j < nl; j++)
				{
					int l1 = i + 1, l2 = j + 1;
					z_av += Vec(i, e1) * Vec(j, e2) * Zmed(l1, l2);
				}
			}
			os = m_dot / h2m * (E[e2] - E[e1]) * pow(z_av, 2.);
			file << "E=" << setw(3) << e1 + 1 << " --> E=" << setw(3) << e2 + 1
					<< ", F=" << setw(8) << setprecision(4) << os << ", ∆E="
					<< setw(7) << setprecision(2) << (E[e2] - E[e1]) * 1000
					<< " meV" << endl;
		}
	}
	file.close();
}
//---------------------------------------------------------------------------

