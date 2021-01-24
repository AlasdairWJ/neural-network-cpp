#pragma once

#include <cstdio>

// Linear Algebra types

namespace la
{

template <size_t N, size_t M>
struct matrix;

template <size_t N>
struct vector
{
	static_assert(N != 0);

	static const size_t Count = N;

	using This = vector<N>;
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

	double &at(const size_t n) { return values[n]; }
	const double &at(const size_t n) const { return values[n]; }

	double &operator[](const size_t n) { return values[n]; }
	const double &operator[](const size_t n) const { return values[n]; }

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

	double* data() { return &values[0]; }
	const double* data() const { return &values[0]; }

	template <size_t O>
	vector<N-O>& offset()
	{
		return *reinterpret_cast<vector<N-O>*>(&values[O]);
	}

	template <size_t O>
	const vector<N-O>& offset() const
	{
		return *reinterpret_cast<const vector<N-O>*>(&values[O]);
	}

	template <size_t L>
	vector<L>& truncate()
	{
		return *reinterpret_cast<vector<L>*>(this);
	}

	template <size_t L>
	const vector<L>& truncate() const
	{
		return *reinterpret_cast<const vector<L>*>(this);
	}

	template <size_t O, size_t L>
	vector<L>& slice() { return offset<O>().truncate<L>(); }

	template <size_t O, size_t L>
	const vector<L>& slice() const { return offset<O>().truncate<L>(); }

	template <size_t Rows, size_t Cols>
	matrix<Rows, Cols>& ravel()
	{
		static_assert(Rows*Cols == N);
		return *reinterpret_cast<matrix<Rows, Cols>*>(this);
	}

	template <size_t Rows, size_t Cols>
	const matrix<Rows, Cols>& ravel() const
	{
		static_assert(Rows*Cols == N);
		return *reinterpret_cast<matrix<Rows, Cols>*>(this);
	}

	matrix<N, 1>& as_column() { return *reinterpret_cast<matrix<N, 1>*>(this); }
	const matrix<N, 1>& as_column() const { return *reinterpret_cast<matrix<N, 1>*>(this); }

	matrix<1, N>& as_row() { return *reinterpret_cast<matrix<1, N>*>(this); }
	const matrix<1, N>& as_row() const { return *reinterpret_cast<matrix<1, N>*>(this); }

private:
	double values[N];
};

// -----------------------------------------------------------------------------

template <size_t N, size_t M>
struct matrix
{
	static_assert(N != 0);
	static_assert(M != 0);

	static const size_t Count = N*M;

	static const size_t Rows = N;
	static const size_t Columns = M;

	using This = matrix<N, M>;
	using RawType = double;

	vector<M>* begin() { return &values[0]; }
	vector<M>* end() { return &values[0] + N; }

	const vector<M>* begin() const { return &values[0]; }
	const vector<M>* end() const { return &values[0] + N; }

	This& zero()
	{
		for (auto& row : *this)
			row.zero();
		return *this;
	}

	vector<M> &at(const size_t n) { return values[n]; }
	const vector<M> &at(const size_t n) const { return values[n]; }

	double &at(const size_t n, const size_t m) { return values[n][m]; }
	const double &at(const size_t n, const size_t m) const { return values[n][m]; }

	vector<M> &operator[](const size_t n) { return values[n]; }
	const vector<M> &operator[](const size_t n) const { return values[n]; }

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

	RawType &raw() { return values; }
	const RawType &raw() const { return values; }

	double* data() { return values[0].data(); }
	const double* data() const { return values[0].data(); }

	vector<Count>& unravel()
	{
		return *reinterpret_cast<vector<Count>*>(this);
	}

	const vector<Count>& unravel() const
	{
		return *reinterpret_cast<const vector<Count>*>(this);
	}

private:
	vector<M> values[N];
};

// -----------------------------------------------------------------------------

template <size_t N, size_t M>
auto product(vector<M> &result, const vector<N> &lhs, const matrix<N, M> &rhs) -> decltype(result)
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
auto product(vector<N> &result, const matrix<N, M> &lhs, const vector<M> &rhs) -> decltype(result)
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
auto product(matrix<I, J> &result, const matrix<I, K> &lhs, const matrix<K, J> &rhs) -> decltype(result)
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
auto print(const vector<N> &x, const char* fmt = "% 8.4lf")
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
auto print(const matrix<N, M> &x, const char* fmt = "% 8.4lf")
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