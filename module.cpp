#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <immintrin.h>
#include <cmath>
// Uncomment for ISPC
//#include "module_ispc.h"
//using namespace ispc;

// ------------------------------------ //
// 	WARM-UP: ACCESSING TENSORS      //
// ------------------------------------ //

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(std::vector<float> &tensor, int &x, int &y, const int &sizeX) {
    // Note that sizeX is the size of a Row, not the number of rows
    return tensor[x * (sizeX)+ y];
}

inline void twoDimWrite(std::vector<float> &tensor, int &x, int &y, const int &sizeX, float &val) {
    tensor[x * (sizeX) + y] = val;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ) {
    return tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * (sizeZ) + b];
}

inline void fourDimWrite(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ, float &val) {
    tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * (sizeZ) + b] = val;
    return; 
}

// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor) {
    tensor = tensor.flatten();
    tensor = tensor.contiguous();
    std::vector<float> vec(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    return vec;
}

/* Programming Your Attention Modules.
 * 
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We have also created O and QK^t Tensors 
 * that are formatted as vectors. After you have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: B, H, N, d:
 *
 * B (Batch Size) - The number of samples for your attention layer. Think of it this way - if I asked my dnn
 * a question and it output 5 different answers it had a batch size of 5. These samples are independent of each
 * other and thus can be parallelized.
 *
 * H (Number of Heads) - Each head runs on its own set of Q, K, V matrices. This effectively allows each head
 * to operate the same attention algorithm, but each with each head using different hyperparameters. These
 * allow each head to have their own definition of what relevance is when looking at a token. These heads
 * can operate independently of one another and thus can be parallized.
 *
 * N (Sequence Length) - The number of tokens. You may think of this as the number of words in a sample.
 *
 * d (Embedding Dimensionality) - The number of features each token encodes per attention head. Let's
 * say I encoded a word using the follow (length, number of vowels, has a capital letters). The
 * emvedded dimensionaliy would be 3.
 * */

// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //

torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)
    
    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);
    
    /* Here is an example of how to read/write 0's to  Q (B, H, N, d) using the 4D accessors

        //loop over Batch Size
         for (int b = 0; b < B; b++) {

             //loop over Heads
             for (int h = 0; h < H; h++) {

                 //loop over Sequence Length
                 for (int i = 0; i < N; i++) {

                     //loop over Embedding Dimensionality
                     for (int j = 0; j < d; j++) {
                        float val = fourDimRead(Q, b, h, i, j, H, N, d);
                        val = 0.0;
                        fourDimWrite(Q, b, h, i, j, H, N, d, val);
                     }
                 }
             }
         }
    */

    /* Here is an example of how to read/write 0's to  QK_t (N, N) using the 2D accessors

           for (int i = 0; i < N; i++) {
	       for (int j = 0; j < N; j++) {
	           float val = twoDimRead(QK_t, i, j, N);
               val = 0.0;
	           twoDimWrite(QK_t, i, j, N, val);
             }
         }
    */
    
    // -------- YOUR CODE HERE  -------- //
     for (int b = 0; b < B; b++) {

             //loop over Heads
             for (int h = 0; h < H; h++) {

                 //loop over Sequence Length
                for (int i = 0; i < N; i++) {
                     //loop over Embedding Dimensionality
                    for (int j = 0; j < N; j++) {
                        // loop over
                        float val = 0.f;
                        for( int k = 0; k < d; k++){
                            val += fourDimRead(Q,b,h,i,k,H,N,d) * fourDimRead(K,b,h,j,k,H,N,d);
                        }
                        twoDimWrite(QK_t,i,j,N,val);
                    }
                }
                for (int i = 0; i < N; ++i){
                    float sum = 0.f;
                    for(int j = 0;j<N;++j){
                        float tmp =  twoDimRead(QK_t,i,j,N);
                        tmp = exp(tmp);
                        sum += tmp;
                        twoDimWrite(QK_t,i,j,N,tmp);
                    }
                    for(int j = 0;j < N;++j){
                        float tmp =  twoDimRead(QK_t,i,j,N) / sum;
                        twoDimWrite(QK_t,i,j,N,tmp);
                    }
                }
                for (int i = 0; i < N; i++) {
                     //loop over Embedding Dimensionality
                    for( int k = 0; k < N; k++){
                        for (int j = 0; j < d; j++) {
                            float val = fourDimRead(O,b,h,i,j,H,N,d);
                        // loop over                       
                            val += twoDimRead(QK_t,i,k,N) * fourDimRead(V,b,h,k,j,H,N,d);
                            fourDimWrite(O,b,h,i,j,H,N,d,val);
                        }
                    }
                }
             }
         }
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //

torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){
    
    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)

    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);
    int TILE = 32;
    // int TILE_2 = 
    // -------- YOUR CODE HERE  -------- //
    for(int b = 0 ; b < B;++b){

        for(int h =0;h<H;++h){

            for(int ti = 0; ti < N; ti += TILE){

                for(int tj = 0; tj < N; tj += TILE){

                    for(int tk = 0; tk < d; tk += TILE){

                        for(int i = ti; i<std::min(N,ti+TILE); ++i){

                            for(int j  = tj; j<std::min(N,tj+TILE); ++j){

                                float val = twoDimRead(QK_t,i,j,N);

                                for(int k = tk; k<std::min(d,tk+TILE); ++k){
                                    val += fourDimRead(Q,b,h,i,k,H,N,d)*fourDimRead(K,b,h,j,k,H,N,d);
                                }
                                twoDimWrite(QK_t,i,j,N,val);
                            }
                        }
                    }
                }
            }
            for (int i = 0; i < N; ++i){
                float sum = 0.f;
                for(int j = 0;j<N;++j){
                    float tmp =  twoDimRead(QK_t,i,j,N);
                    tmp = exp(tmp);
                    sum += tmp;
                    twoDimWrite(QK_t,i,j,N,tmp);
                }
                for(int j = 0;j < N;++j){
                    float tmp =  twoDimRead(QK_t,i,j,N) / sum;
                    twoDimWrite(QK_t,i,j,N,tmp);
                }
            }
            for(int ti = 0; ti < N; ti += TILE){

                for(int tk = 0; tk < N; tk += TILE){

                    for(int tj = 0; tj < d; tj += TILE){

                        for(int i = ti; i<std::min(N,ti+TILE); ++i){

                            for(int k = tk; k<std::min(N,tk+TILE); ++k){

                                for(int j  = tj; j<std::min(d,tj+TILE); ++j){
                                    
                                    float val = fourDimRead(O,b,h,i,j,H,N,d);
                                    val += twoDimRead(QK_t,i,k,N)*fourDimRead(V,b,h,k,j,H,N,d);
                                    fourDimWrite(O,b,h,i,j,H,N,d,val);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // // DO NOT EDIT THIS RETURN STATEMENT //
    // // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION     	              //
// ---------------------------------------------------------- //

torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor temp,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)

    //Make O Tensor with Shape (B, H, N, d)
    //and O Row Tensor with Shape (N)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
    at::Tensor ORowTensor = at::zeros({N}, at::kFloat);

    //Format Y, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    
    //Format ORow Tensor into a 1D vector
    // You can simply access this as ORow[i]
    std::vector<float> ORow = formatTensor(ORowTensor);


    // -------- YOUR CODE HERE  -------- //
    // We give you a template of the first three loops for your convenience
    //loop over batch
    // #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++){

        //loop over heads
        for (int h = 0; h < H; h++){
            for (int i = 0; i < N ; i++){
		// YRow is moved inside so each OpenMP thread gets a local copy.
                at::Tensor ORowTensor = temp.index({torch::indexing::Slice(omp_get_thread_num(), torch::indexing::None)});      
                std::vector<float> ORow = formatTensor(ORowTensor);
		//YOUR CODE HERE
                
                for(int j = 0; j < N; ++j){
                    float val = 0.f;
                    for(int k = 0; k < d; ++k){
                        val += fourDimRead(Q,b,h,i,k,H,N,d) * fourDimRead(K, b, h, j, k, H, N, d);
                    }
                    ORow[j] = val;
                }
                float sum = 0.f;
                for(int j =0; j < N; ++j){
                    float val = exp(ORow[j]);
                    ORow[j] = val;
                    sum += val;
                }
                for(int j =0; j < N; ++j){
                    ORow[j] /=  sum;
                }
                for(int k = 0 ; k < N; ++k){
                    for(int j = 0; j < d ; ++j){
                        float val = fourDimRead(O,b,h,i,j,H,N,d);
                        val += ORow[k] * fourDimRead(V,b,h,k,j,H,N,d);
                        fourDimWrite(O,b,h,i,j,H,N,d,val);    
                    }
                }
            }
	}
    }
	    
	
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION 		      //
// ---------------------------------------------------------- //

torch::Tensor myFlashAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor,
               torch::Tensor QiTensor, torch::Tensor KjTensor, torch::Tensor VjTensor,
               torch::Tensor SijTensor, torch::Tensor PijTensor, torch::Tensor PVTensor,
               torch::Tensor OiTensor, torch::Tensor LTensor,  torch::Tensor LiTensor, 
	       torch::Tensor LijTensor, torch::Tensor LnewTensor, int Bc, int Br,
                int B, int H, int N, int d) {
        
    // Q, K, V are passed in with Shape: (B, H, N, d)
    // Sij, Pij are passed in with Shape: (Br, Bc)
    // Kj, Vj are passed in with Shape: (Bc, d)
    // Qi, Oi, and PV  are passed in with Shape: (Br, d)
    // L in passed in with Shape: (N)
    // Li, Lij, and Lnew are passed in with shape (Br)

    //Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
   
    //Format All Tensors into Vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    std::vector<float> Sij = formatTensor(SijTensor);
    std::vector<float> Pij = formatTensor(PijTensor);
    std::vector<float> Kj = formatTensor(KjTensor);
    std::vector<float> Vj = formatTensor(VjTensor);
    std::vector<float> Qi = formatTensor(QiTensor);
    std::vector<float> Oi = formatTensor(OiTensor);
    std::vector<float> l = formatTensor(LTensor);
    std::vector<float> PV = formatTensor(PVTensor);
    std::vector<float> li = formatTensor(LiTensor);
    std::vector<float> lij = formatTensor(LijTensor);
    std::vector<float> lnew = formatTensor(LnewTensor);

    // -------- YOUR CODE HERE  -------- //
     for (int b = 0; b < B; b++)
    {
        // loop over heads
        for (int h = 0; h < H; h++)
        {
            for (int tj = 0; tj < N; tj += Bc)
            {
                for (int ti = 0; ti < N; ti += Br)
                {
                    // calculate exp(QK) for block (Br, Bc)
                    for (int i = ti; i < std::min(ti + Br, N); i++)
                        {
                            int ii = i - ti;

                        for (int j = tj; j < std::min(tj + Bc, N); j++)
                        {
                        int jj = j - tj;
                        
                            float val = 0.0;
                            for (int k = 0; k < d; k++)
                            {
                                float qVal = fourDimRead(Q, b, h, i, k, H, N, d);
                                float kVal = fourDimRead(K, b, h, j, k, H, N, d);
                                val += qVal * kVal;
                            }
                            val = exp(val);
                            twoDimWrite(Pij, ii, jj, Bc, val);
                        }
                    }

                    // apply blocked softmax
                    for (int i = ti; i < std::min(ti + Br, N); i++)
                    {
                        int ii = i - ti;
                        float rowSum = 0.0;
                        for (int j = tj; j < std::min(tj + Bc, N); j++)
                        {
                            int jj = j - tj;
                            rowSum += twoDimRead(Pij, ii, jj, Bc);
                        }
                        float prevSum = l[i];
                        float newSum = prevSum + rowSum;
                        l[i] = newSum;
                        for (int k = 0; k < d; k++)
                        {
                            float pv = 0.0;
                            float oVal = fourDimRead(O, b, h, i, k, H, N, d);
                            for (int j = tj; j < std::min(tj + Bc, N); j++)
                            {
                                int jj = j - tj;
                                float pVal = twoDimRead(Pij, ii, jj, Bc);
                                float vVal = fourDimRead(V, b, h, j, k, H, N, d);
                                pv += pVal * vVal;
                            }
                            oVal = (prevSum * oVal + pv) / newSum;
                            fourDimWrite(O, b, h, i, k, H, N, d, oVal);
                        }
                    }
                }
            }
            std::fill(l.begin(), l.end(), 0);
        }
    }
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
  m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked, " Blocked Unfused Attention");
  m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
  m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
  m.def("twoDimRead", &twoDimRead, "twoDimRead");
  m.def("fourDimRead", &fourDimRead, "fourDimRead");
}
