

#define API __declspec(dllexport)

API void add(int N, /*[out]*/ double *out, double *a1, double *a2)
{
    int i;
    for (i=0; i<N; i++)
        out[i] = a1[i] + a2[i];
}
