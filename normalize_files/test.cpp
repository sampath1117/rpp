#include<iostream>
#include<cmath>
using namespace std;


void fill_input(int *srcPtr, int volume, int mulFactor, int addFactor)
{
    for(int i = 0; i < volume; i++)
        srcPtr[i] = i * mulFactor + addFactor;
}

void display(int *srcPtr, int *roi)
{
    for(int i = 0; i < roi[0]; i++)
    {
        for(int j = 0; j < roi[1]; j++)
        {
            for(int k = 0; k < roi[2]; k++)
            {
                int index = i * roi[1] * roi[2] + j * roi[2] + k;
                cout << srcPtr[index] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
}

int compute_param_index(int i, int j, int k, int *paramShape, int *paramStrides)
{

    int factor0 =  (paramShape[0] > 1) ? (i % paramShape[0]) * paramStrides[0] : 0;
    int factor1 =  (paramShape[1] > 1) ? (j % paramShape[1]) * paramStrides[1] : 0;
    int factor2 =  (paramShape[2] > 1) ? (k % paramShape[2]) * paramStrides[2] : 0;
    int paramIndex = factor0 + factor1 + factor2;
    return paramIndex;
}

void compute_normalization(int *srcPtr, int *dstPtr, int *mean, int *roi, int *paramShape, int *paramStrides)
{
    for(int i = 0; i < roi[0]; i++)
    {
        for(int j = 0; j < roi[1]; j++)
        {
            for(int k = 0; k < roi[2]; k++)
            {
                int index = i * roi[1] * roi[2] + j * roi[2] + k;
                int paramIndex = compute_param_index(i, j, k , paramShape, paramStrides);
                dstPtr[index] = srcPtr[index] - mean[paramIndex];
                cout << "i, j, k, meanIndex: "<< "(" << i << ", " << j << ", " << k << "): " <<paramIndex << endl;
            }
        }
    }

}

int main()
{
    int nDim = 3;
    int *roi = new int[nDim];
    int volume = 1;
    int paramVolume = 1;
    cout << "ROI values: ";
    for(int i = 0; i < nDim; i++)
    {
        roi[i] = i + 2;
        volume *= roi[i];
        cout << roi[i] << " ";
    }
    cout << endl;

    int axisMask;
    cout << "Enter axisMask value: " << endl;
    cin >> axisMask;

    cout << "paramShape values: " << endl;
    int *paramShape = new int[nDim];
    for(int i = 0; i < nDim; i++)
    {
        paramShape[i] = ((axisMask & (int)(pow(2, i))) >= 1) ? 1 : roi[i];
        paramVolume *= paramShape[i];
        cout << paramShape[i] << " ";
    }
    cout << endl;

    int *paramStrides = new int[nDim];
    int val = 1;
    for(int i = nDim - 1; i > 0; i--)
    {
        paramStrides[i] = val;
        val *= paramShape[i];
    }
    paramStrides[0] = val;

    cout << "paramStride values: " << endl;
    for(int i = 0; i < nDim; i++)
        cout << paramStrides[i] << " ";

    cout << endl;

    // Allocate memory for input
    int *srcPtr = new int[volume];
    fill_input(srcPtr, volume, 1, 1);

    int *mean = new int[paramVolume];
    fill_input(mean, paramVolume, 5, 0);

    int *dstPtr = new int[volume];
    compute_normalization(srcPtr, dstPtr, mean, roi, paramShape, paramStrides);

    cout << "printing input: " << endl;
    display(srcPtr, roi);

    cout << "printing mean: " << endl;
    display(mean, paramShape);

    cout << "printing output: " << endl;
    display(dstPtr, roi);

    delete[] roi;
    delete[] paramShape;
    delete[] paramStrides;
    delete[] srcPtr;
    delete[] mean;
    delete[] dstPtr;
    return 0;
}

