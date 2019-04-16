#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <mpi.h>

#define N 1000000 //particles
#define Lx 100 //length
#define CELLS_NUM 10001 //cells

#define PI 3.14159265358979323846f

#define Q(k) ((k) % 2 ? -1 : 1)
#define M(k) ((k) % 2 ? 1 : 1836)

#define E(x) sin(2.0f * PI / CELLS_NUM * (x))

#define INITIAL_VELOCITY 0.0f
#define TIME_INTERVAL 0.01f
#define TIME_STEPS 50

typedef unsigned long long ull;

struct Particle
{
	float coord[2],
		velocity[2];
	short mass;
	char charge;
};

float distributionFunction(ull particleNumber)
{
	return (float)particleNumber * Lx / (float)N;
}

ull initialize(Particle *particles, float lower, float upper)
{
	ull current = 0;
	for (ull i = 0; i < N; i++)
	{
		float coord = distributionFunction(i);
		if (coord >= lower && coord < upper)
		{
			particles[current].coord[0] = coord;
			particles[current].coord[1] = coord;
			particles[current].velocity[0] = INITIAL_VELOCITY;
			particles[current].velocity[1] = INITIAL_VELOCITY;
			particles[current].mass = M(i);
			particles[current].charge = Q(i);
			current++;
		}
	}
	
	for (ull i = current; i < N; i++)
	{
		particles[i].mass = 0;
	}

	return current;
}

struct ProcessInfo
{
	const int rank,
		size;
	
	size_t coef;

	float *density,
		*charge,
		lower,
		upper,
		cellSize;

	Particle *particles,
		*sendRight,
		*sendLeft;

	char period = 1;
	
	ull *nums,
		particleNum,
		firstNull,
		sRSize,
		sLSize;

	ProcessInfo(int r, int s) : rank{ r }, size{ s }
	{
		coef = (CELLS_NUM - 1) / size;

		nums = new ull[coef];

		density = new float[2 * (coef + (rank == size - 1))];
		charge = density + coef;

		particles = new Particle[3 * N];
		sendRight = particles + N;
		sendLeft = sendRight + N;

		cellSize = (float)Lx / CELLS_NUM;
		lower = rank * coef * cellSize;
		upper = (rank + 1) * coef * cellSize;

		particleNum = initialize(particles, lower, upper);
		firstNull = particleNum;
		sRSize = 0ull;
		sLSize = 0ull;
	}

	~ProcessInfo()
	{
		delete[] nums;
		delete[] density;
		delete[] particles;
	}

	void newIteration()
	{
		sRSize = 0;
		sLSize = 0;
		period ^= 1;

		for (size_t c = 0; c < coef; c++)
		{
			density[c] = 0.0f;
			nums[c] = 0;
		}
	}

	void moveParticle(ull p)
	{
		particles[p].coord[period] = particles[p].coord[period ^ 1]
			+ particles[p].velocity[period] * TIME_INTERVAL;

		particles[p].velocity[period] = particles[p].velocity[period ^ 1]
			+ 2.0f * particles[p].charge * E(particles[p].coord[period]) / particles[p].mass;
	}

	void sumCharge(float coord, char pcharge)
	{
		float index = coord / cellSize;

		size_t i = static_cast<size_t>(index) + (index - (int)index > 0.5f);
		i -= rank * coef;

		if (i >= coef) i--;

		charge[i] += (float)pcharge;
		nums[i]++;
	}

	char isInBorders(ull p)
	{
		return (particles[p].coord[period] >= lower) - (particles[p].coord[period] < upper);
	}

	void checkNullAndNum(ull p)
	{
		if (firstNull > p)
			firstNull = p;

		if (p == particleNum - 1)
		{
			while (!particles[particleNum].mass && particleNum)
			{
				particleNum--;
			}
			particleNum++;
		}
	}

	void writeRight(ull p)
	{
		sendRight[sRSize++] = particles[p];
		particles[p].mass = 0;

		checkNullAndNum(p);
	}

	void writeLeft(ull p)
	{
		sendLeft[sLSize++] = particles[p];
		particles[p].mass = 0;

		checkNullAndNum(p);
	}

	void placeParticles(Particle *rBuf, ull rSize)
	{
		for (size_t i = 0; i < rSize; i++)
		{
			sumCharge(rBuf[i].coord[period], rBuf[i].charge);
		}

		for (size_t i = 0; i < rSize; i++)
		{
			while (particles[firstNull].mass)
			{
				firstNull++;
			}
			particles[firstNull++] = rBuf[i];
		}
		if (firstNull > particleNum)
			particleNum = firstNull;
	}

	void exchangeParticles(MPI_Datatype particleType)
	{
		ull rRSize,
			rLSize;

		MPI_Status stat[4];
		MPI_Request req[4];

		MPI_Isend(&sRSize, 1, MPI_UNSIGNED_LONG_LONG, (rank + 1) % size, 4, MPI_COMM_WORLD, req);
		MPI_Isend(&sLSize, 1, MPI_UNSIGNED_LONG_LONG, (rank - 1 + size) % size, 7, MPI_COMM_WORLD, req + 1);

		MPI_Irecv(&rRSize, 1, MPI_UNSIGNED_LONG_LONG, (rank + 1) % size, 7, MPI_COMM_WORLD, req + 2);
		MPI_Irecv(&rLSize, 1, MPI_UNSIGNED_LONG_LONG, (rank - 1 + size) % size, 4, MPI_COMM_WORLD, req + 3);
		MPI_Waitall(4, req, stat);

		Particle *rRBuf = new Particle[static_cast<unsigned>(rRSize + rLSize)],
			*rLBuf = rRBuf + rRSize;

		MPI_Isend(sendRight, static_cast<int>(sRSize), particleType,
			(rank + 1) % size, 141, MPI_COMM_WORLD, req);

		MPI_Isend(sendLeft, static_cast<int>(sLSize), particleType,
			(rank - 1 + size) % size, 171, MPI_COMM_WORLD, req + 1);

		MPI_Irecv(rRBuf, static_cast<int>(rRSize), particleType,
			(rank + 1) % size, 171, MPI_COMM_WORLD, req + 2);

		MPI_Irecv(rLBuf, static_cast<int>(rLSize), particleType,
			(rank - 1 + size) % size, 141, MPI_COMM_WORLD, req + 3);

		MPI_Waitall(4, req, stat);

		placeParticles(rRBuf, rRSize + rLSize);

		delete[] rRBuf;
	}

	void calculateDensity()
	{
		for (size_t c = 0; c < coef; c++)
		{
			density[c] = charge[c] / (nums[c] ? nums[c] : 1);
		}
	}
};

void createType(MPI_Datatype *particleType)
{
	const int nitems = 4;
	int blocklengths[] = { 2, 2, 1, 1 };
	MPI_Datatype types[] = { MPI_FLOAT, MPI_FLOAT, MPI_SHORT, MPI_CHAR };
	MPI_Aint offsets[4];

	offsets[0] = offsetof(Particle, coord);
	offsets[1] = offsetof(Particle, velocity);
	offsets[2] = offsetof(Particle, mass);
	offsets[3] = offsetof(Particle, charge);

	MPI_Type_create_struct(nitems, blocklengths, offsets, types, particleType);
	MPI_Type_commit(particleType);
}

void printCoords(ProcessInfo &pinfo, MPI_Datatype particleType, size_t ts)
{
	int *displs = nullptr,
		*rcounts = nullptr;

	if (pinfo.rank == 0)
	{
		displs = new int[pinfo.size * 2];
		rcounts = displs + pinfo.size;
	}

	int toSend = static_cast<int>(pinfo.particleNum);
	MPI_Gather(&toSend, 1, MPI_INT, rcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

	ull count = 0;
	if (pinfo.rank == 0)
	{
		for (int i = 0; i < pinfo.size; i++)
		{
			displs[i] = static_cast<int>(count);
			count += rcounts[i];
		}
	}

	Particle *allParts = nullptr;
	if (pinfo.rank == 0)
		allParts = new Particle[static_cast<unsigned>(count)];

	MPI_Gatherv(pinfo.particles, static_cast<int>(pinfo.particleNum), particleType,
		allParts, rcounts, displs, particleType, 0, MPI_COMM_WORLD);

	if (pinfo.rank == 0)
	{
		std::string coordfile = "coord" + std::to_string(ts);
		freopen(coordfile.c_str(), "w", stdout);

		for (size_t p = 0; p < count; p++)
		{
			if (allParts[p].mass)
				printf("%f %f\n", allParts[p].coord[pinfo.period],
					E(allParts[p].coord[pinfo.period]));
		}

		delete[] displs;
		delete[] allParts;
	}
}

void printDens(ProcessInfo &pinfo, size_t ts)
{
	float *allDensity = nullptr;

	if (pinfo.rank == 0)
	{
		allDensity = new float[CELLS_NUM];
	}

	MPI_Gather(pinfo.density, pinfo.coef, MPI_FLOAT,
		allDensity, pinfo.coef, MPI_FLOAT, 0, MPI_COMM_WORLD);
	
	if (pinfo.rank == pinfo.size - 1)
	{
		MPI_Send(pinfo.density + pinfo.coef, 1, MPI_FLOAT, 0, 504, MPI_COMM_WORLD);
	}

	if (pinfo.rank == 0)
	{
		MPI_Recv(allDensity + CELLS_NUM - 1, 1, MPI_FLOAT, pinfo.size - 1, 504,
			MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		std::string densfile = "dens" + std::to_string(ts);
		freopen(densfile.c_str(), "w", stdout);

		for (size_t c = 0; c < CELLS_NUM; c++)
		{
			printf("%f\n", allDensity[c]);
		}

		delete[] allDensity;
	}
}

int main(int argc, char **argv)
{
	int size, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	ProcessInfo pinfo{ rank, size };

	MPI_Datatype particleType;
	createType(&particleType);

	for (size_t ts = 0; ts < TIME_STEPS; ts++)
	{
		pinfo.newIteration();

		for (ull p = 0; p < pinfo.particleNum; p++)
		{
			pinfo.moveParticle(p);

			switch (pinfo.isInBorders(p))
			{
			case 0:
				pinfo.sumCharge(pinfo.particles[p].coord[pinfo.period], pinfo.particles[p].charge);
				break;

			case 1:
				pinfo.writeRight(p);
				break;

			default:
				pinfo.writeLeft(p);
			}
			
		}

		pinfo.exchangeParticles(particleType);

		printCoords(pinfo, particleType, ts);

		pinfo.calculateDensity();

		printDens(pinfo, ts);
	}

	MPI_Finalize();
	return 0;
}