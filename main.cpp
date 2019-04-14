#include <iostream>
#include <math.h>

#define N 1000000 //particles
#define Lx 100 //length
#define Nx 10000 //cells

#define PI 3.14159265358979323846

#define Q(k) ((k) % 2 ? -1 : 1)
#define M(k) ((k) % 2 ? 1 : 1836)

#define E(x) sin(2.0 * PI /(Nx * (x)))

struct Particle
{
	float coord,
		velocity;
	short mass;
	char charge;
};

float distributionFunction(size_t particleNumber)
{
	return (float)particleNumber * Lx / (float)N;
}

void initialize(Particle *particles)
{
	for (size_t i = 0; i < N; i++)
	{
		particles[i].coord = distributionFunction(i);
		particles[i].velocity = 0.0f;
		particles[i].mass = M(i);
		particles[i].charge = Q(i);
	}
}
 
int main(int argc, char **argv)
{
	Particle *particles = new Particle[N];
	
	initialize(particles);


	float *electricField = new float[4 * Nx],
		*density = electricField + Nx,
		*charge = density + Nx,
		*phi = charge + Nx,
		cellSize = (float)Lx / (Nx - 1), //cell size
		dt = 0.01f; //time interval
	
	for (size_t c = 0; c < N; c++)
	{
		phi[c] = 0.0f;
	}

	for (size_t ts = 0; ts < 50; ts++)
	{
		//STEP 1: Compute density
		for (size_t c = 0; c < N; c++)
		{
			density[c] = 0.0f;
			electricField[c] = 0.0f;
		}

		for (size_t p = 0; p < N; p++)
		{
			float fi = particles[p].coord / cellSize - 1.0f;
			int i = (int)fi;
			float rem = fi - i;

			charge[i] += (float)particles[p].charge * (1 - rem);
			charge[i + 1] += (float)particles[p].charge * rem;
		}

		for (size_t c = 0; c < N; c++)
		{
			density[c] = (c == 0 || c == Nx - 1 ? 4.0f : 2.0f) * charge[c] / cellSize * cellSize + 1e4;
		}
		
		//STEP 2: Compute potential

		//solving puasson into phi array

		//STEP 3: Compute electric field
		

		for (size_t i = 1; i < Nx - 1; i++)
		{
			electricField[i] = phi[i - 1] - phi[i + 1];
		}
		
		electricField[0] = 2.0f * (phi[0] - phi[2]);
		electricField[Nx - 1] = 2.0f * (phi[Nx - 2] - phi[Nx - 1]);
		
		for (size_t i = 1; i < Nx - 1; i++)
		{
			electricField[i] /= 2.0f * cellSize;
		}

		//STEP 4: Move particles
		float QE = 1.602e-19;
		for (size_t p = 0; p < N; p++)
		{
			float fi = particles[p].coord / cellSize - 1.0f;
			int i = (int)fi;
			float rem = fi - i;

			float E = electricField[i] * (1 - rem) + electricField[i + 1] * rem;
			
			particles[p].coord += particles[p].velocity * dt;
		}
	}

	delete[] electricField;
	delete[] particles;
	return 0;
}