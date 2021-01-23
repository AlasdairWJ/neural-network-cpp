#pragma once

#include <cstdio>

// Linear Algebra types

namespace la
{

template <size_t N>
struct Vector
{
	static_assert(N != 0);

	static const size_t Count = N;

	using This = Vector<N>;
	using RawType = double[N];

	double* begin() { return &values[0]; }
	double* end() { return &values[0] + N; }

	const double* begin() const { return &values[0]; }
	const double* end() const { return &values[0] + N; }

	This& zero()
	{
		for (double& v : values)
			v = 0.0;
		return *this;
	}

	double &operator[](const size_t n)
	{
		return values[n];
	}

	const double &operator[](const size_t n) const
	{
		return values[n];
	}

	This &operator+=(const This& other)
	{
		for (size_t n = 0; n < N; n++)
			values[n] += other.values[n];
		return *this;
	}

	This &operator-=(const This& other)
	{
		for (size_t n = 0; n < N; n++)
			values[n] -= other.values[n];
		return *this;
	}

	This &operator*=(const double factor)
	{
		for (size_t n = 0; n < N; n++)
			values[n] *= factor;
		return *this;
	}

	This &operator/=(const double factor)
	{
		for (size_t n = 0; n < N; n++)
			values[n] /= factor;
		return *this;
	}

	This &operator==(const This& other)
	{
		for (size_t n = 0; n < N; n++)
			if (values[n] != other.values[n])
				return false;
		return true;
	}

	This &operator!=(const This& other)
	{
		return !operator==(other);
	}

	RawType &raw() { return values; }
	const RawType &raw() const { return values; }

	template <size_t O>
	Vector<N-O>& offset()
	{
		return *reinterpret_cast<Vector<N-O>*>(&values[O]);
	}

	template <size_t O>
	const Vector<N-O>& offset() const
	{
		return *reinterpret_cast<const Vector<N-O>*>(&values[O]);
	}

	template <size_t L>
	Vector<L>& truncate()
	{
		return *reinterpret_cast<Vector<L>*>(this);
	}

	template <size_t L>
	const Vector<L>& truncate() const
	{
		return *reinterpret_cast<const Vector<L>*>(this);
	}

private:
	double values[N];
};

// -----------------------------------------------------------------------------

template <size_t N, size_t M>
struct Matrix
{
	static_assert(N != 0);
	static_assert(M != 0);

	static const size_t Count = N*M;

	static const size_t Rows = N;
	static const size_t Columns = M;

	using This = Matrix<N, M>;
	using RawType = double;

	Vector<M>* begin() { return &values[0]; }
	Vector<M>* end() { return &values[0] + N; }

	const Vector<M>* begin() const { return &values[0]; }
	const Vector<M>* end() const { return &values[0] + N; }

	This& zero()
	{
		for (auto& row : *this)
			row.zero();
		return *this;
	}

	Vector<M> &operator[](const size_t n)
	{
		return values[n];
	}

	const Vector<M> &operator[](const size_t n) const
	{
		return values[n];
	}

	This &operator+=(const This& other)
	{
		for (size_t n = 0; n < N; n++)
			values[n] += other.values[n];
		return *this;
	}

	This &operator-=(const This& other)
	{
		for (size_t n = 0; n < N; n++)
			values[n] -= other.values[n];
		return *this;
	}

	This &operator*=(const double factor)
	{
		for (size_t n = 0; n < N; n++)
			values[n] *= factor;
		return *this;
	}

	This &operator/=(const double factor)
	{
		for (size_t n = 0; n < N; n++)
			values[n] /= factor;
		return *this;
	}

	RawType &data() { return values; }
	const RawType &data() const { return values; }

	Vector<Count>& unravel()
	{
		return *reinterpret_cast<Vector<Count>*>(this);
	}

	const Vector<Count>& unravel() const
	{
		return *reinterpret_cast<const Vector<Count>*>(this);
	}


private:
	Vector<M> values[N];
};

// -----------------------------------------------------------------------------

template <size_t N, size_t M>
auto product(Vector<M> &result, const Vector<N> &lhs, const Matrix<N, M> &rhs) -> decltype(result)
{
	for (size_t m = 0; m < M; m++)
	{
		result[m] = 0.0;
		for (size_t n = 0; n < N; n++)
			result[m] += lhs[n] * rhs[n][m];
	}
	return result;
}

template <size_t N, size_t M>
auto product(Vector<N> &result, const Matrix<N, M> &lhs, const Vector<M> &rhs) -> decltype(result)
{
	for (size_t n = 0; n < N; n++)
	{
		result[n] = 0.0;
		for (size_t m = 0; m < M; m++)
			result[n] += lhs[n][m] * rhs[m];
	}
	return result;
}

template <size_t I, size_t J, size_t K>
auto product(Matrix<I, J> &result, const Matrix<I, K> &lhs, const Matrix<K, J> &rhs) -> decltype(result)
{
	for (size_t i = 0; i < I; i++)
		for (size_t j = 0; j < J; j++)
		{
			result[i][j] = 0.0;
			for (size_t k = 0; k < K; k++)
				result[i][j] += lhs[i][k] * rhs[k][j];
		}
	return result;
}

// -----------------------------------------------------------------------------

template <size_t N>
auto print(const Vector<N> &x, const char* fmt = "% 8.4lf")
{
	for (size_t n = 0; n < N; n++)
	{
		putchar(n == 0 ? '{' : ' ');
		printf(fmt, x[n]);
		printf(", ");
		putchar(n + 1 == N ? '}' : '\n');
	}
	putchar('\n');
}

template <size_t N, size_t M>
auto print(const Matrix<N, M> &x, const char* fmt = "% 8.4lf")
{
	for (size_t n = 0; n < N; n++)
	{
		putchar(n == 0 ? '{' : ' ');
		putchar('{');
		for (size_t m = 0; m < M; m++)
		{
			if (m != 0) printf(", ");
			printf(fmt, x[n][m]);
		}
		putchar('}');
		putchar(n + 1 == N ? '}' : ',');
		putchar('\n');
	}
}

} // la