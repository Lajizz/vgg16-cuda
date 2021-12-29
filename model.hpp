#include "utils.h"
/* basic infomation */
struct Layerinfo{
    std::string type;
    std::string name;
};

/* convolution layer with relu */
struct ConvConf
{
    /* layer infomation */
    Layerinfo info;

    /* attrubutes */
    std::vector<int> dilations;
    unsigned int group;
    std::vector<int> kernel_shape;
    std::vector<int> pads;
    std::vector<int> strides;

    /* data input */
    std::vector<int> input_shape;
    float *input;
    std::vector<int> weight_shape;
    float *weight;
    unsigned bias_shape;
    float *bias;

    /* layer output */
    std::vector<int> output_shape;
    float *output;
};


/* MaxPool layer */
struct MaxPoolConf
{
    /* layer information */
    // Layerinfo info;

    /* attributes */
    std::vector<int> kernel_shape;
    std::vector<int> pads;
    std::vector<int> strides;

    /* data input */
    std::vector<int> input_shape;
    float *input;

    /* intermedia output */
    std::vector<int> output_shape;
    float *output;
};

struct GemmConf{
    /* layer information */
    // Layerinfo info;

    /* attributes */
    unsigned alpha;
    unsigned beta;
    unsigned transB;

    /* data input */
    std::vector<int> input_shape;
    float *input;
    std::vector<int> weight_shape;
    float *weight;
    unsigned bias_shape;
    float *bias;

    /* layer output */
    std::vector<int> output_shape;
    float *output;
};


/* parent class */
class layer {
    public:
        Layerinfo info;
        layer(std::string,std::string);
        void print_message();
        virtual ~layer() = default;
        
};

class convolution :public layer{
    public:
        ConvConf conf;
        convolution(std::string name,std::string type):layer(name,type){};
        void print_message();
        ~convolution();
};

class maxpool :public layer{
    public:
        MaxPoolConf conf;
        maxpool(std::string name,std::string type):layer(name,type){};
        void print_message();
        ~maxpool();
};

class gemm :public layer{
    public:
        GemmConf conf;
        gemm(std::string name,std::string type):layer(name,type){};
        void print_message();
        ~gemm();
};


/* the whole model*/
class VGG16{
    public:
        std::vector<int> input_shape;
        float* input;
        std::vector<int> output_shape;
        float* output;
        std::vector<layer *> paras;
        VGG16(std::string, std::string);
        void print_message();
        void inference(float* in, float* out);
        ~VGG16();
};


/* construct function */
layer::layer(std::string name, std::string type){
    info.name = name;
    info.type = type;
}

/* help function */
void layer::print_message(){
    std::cout<<"---------layer infomation-------------\n";
    std::cout<<"Layer name: "<<info.name<<"\n";
    std::cout<<"Layer type: "<<info.type<<"\n";
}

/* help function */
void convolution::print_message(){
    layer::print_message();
    std::cout<<"input shape: ";
    vector_printf(conf.input_shape);
    std::cout<<"\n";
    std::cout<<"output shape: ";
    vector_printf(conf.output_shape);
    std::cout<<"\n";
    std::cout<<"weight shape: ";
    vector_printf(conf.weight_shape);
    std::cout<<"\n";
    std::cout<<"bias shape: "<<conf.bias_shape<<"\n";
    
}

convolution::~convolution(){
    cudaFree(conf.input);
    cudaFree(conf.output);
    cudaFree(conf.weight);
    cudaFree(conf.bias);
}

/* help function */
void maxpool::print_message(){
    layer::print_message();
    std::cout<<"input shape: ";
    vector_printf(conf.input_shape);
    std::cout<<"\n"; 
    std::cout<<"output shape: ";
    vector_printf(conf.output_shape);
    std::cout<<"\n";
}

maxpool::~maxpool(){
    cudaFree(conf.input);
    cudaFree(conf.output);
}

/* help function */
void gemm::print_message(){
    layer::print_message();
    std::cout<<"input shape: ";
    vector_printf(conf.input_shape);
    std::cout<<"\n";
    std::cout<<"output shape: ";
    vector_printf(conf.output_shape);
    std::cout<<"\n";
    std::cout<<"weight shape: ";
    vector_printf(conf.weight_shape);
    std::cout<<"\n";
    std::cout<<"bias shape: "<<conf.bias_shape<<"\n";
}

gemm::~gemm(){
    cudaFree(conf.input);
    cudaFree(conf.output);
    cudaFree(conf.weight);
    cudaFree(conf.bias);
}

//load parameter and copy to global
VGG16::VGG16(std::string config, std::string paras_dir){
    std::cout << "start to load paras:" << std::endl;
    json para;
    std::ifstream in(config);
    in >> para;
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        exit(EXIT_FAILURE);
    }

    input_shape.push_back(1);
    input_shape.push_back(3);
    input_shape.push_back(244);
    input_shape.push_back(244);
    output_shape.push_back(1);
    output_shape.push_back(1000);


    std::vector<int> data_in(input_shape);
    std::vector<int> data_out(input_shape);
    for (int i = 0; i < 38; i++)
    {
        std::string index = std::to_string(i);
        std::string name = para[index]["name"];
        std::string type = para[index]["type"];
        
        if(type == "Conv"){
            convolution* l = new convolution(name, type);
            // attributes
            std::vector<int> dilations = para[index]["dilations"];
            l->conf.dilations = dilations;
            l->conf.group = para[index]["group"];
            std::vector<int> kernel_shape = para[index]["kernel_shape"];
            l->conf.kernel_shape = kernel_shape;
            std::vector<int> pads = para[index]["pads"];
            l->conf.pads = pads;
            std::vector<int> strides = para[index]["strides"];
            l->conf.strides = strides;
            // weight and bias
            std::vector<int> weight_shape = para[index]["weight_shape"];
            l->conf.weight_shape = weight_shape;
            l->conf.bias_shape =para[index]["bias_shape"];

            int ws = vector_multi(l->conf.weight_shape);
            int bs = l->conf.bias_shape;
            float* src_weight = new float[ws];
            float* src_bias = new float[bs];
            std::string WeightFile = paras_dir + std::string(para[index]["weight"]);
            std::string BiasFile = paras_dir + std::string(para[index]["bias"]);
            
            std::ifstream i(WeightFile, std::ios::in | std::ios::binary);
            i.read((char *)src_weight, ws * sizeof(float));
            i.close();
            std::ifstream j(BiasFile, std::ios::in | std::ios::binary);
            j.read((char *)src_bias, bs * sizeof(float));
            j.close();

            // printf("%f %f %f\n",src_weight[0],src_weight[1],src_weight[2]);
            l->conf.weight = copytogpu(src_weight, l->conf.weight, ws);
            l->conf.bias = copytogpu(src_bias, l->conf.bias, bs);
            delete []src_weight;
            delete []src_bias;

            //input and output
            l->conf.input_shape = data_out;
            data_out[1] = l->conf.weight_shape[0];
            l->conf.input = gpumalloc(l->conf.input,vector_multi(l->conf.input_shape));
            l->conf.output_shape = data_out;
            l->conf.output = gpumalloc(l->conf.output,vector_multi(l->conf.output_shape));
            
            paras.push_back((layer *)l);
        }else if(type == "MaxPool"){
            maxpool *l = new maxpool(name, type);
            // attribute
            std::vector<int> kernel_shape = para[index]["kernel_shape"];
            l->conf.kernel_shape = kernel_shape;
            std::vector<int> pads = para[index]["pads"];
            l->conf.pads = pads;
            std::vector<int> strides = para[index]["strides"];
            l->conf.strides = strides;
            // input and output
            l->conf.input_shape = data_out;
            data_out[2] /= 2;
            data_out[3] /= 2;
            l->conf.input = gpumalloc(l->conf.input,vector_multi(l->conf.input_shape));
            l->conf.output_shape = data_out;
            l->conf.output = gpumalloc(l->conf.output,vector_multi(l->conf.output_shape));

            paras.push_back((layer *)l);
        }else if(type == "Gemm"){
            gemm * l = new gemm(name, type);
            // attributes
            l->conf.alpha = para[index]["alpha"];
            l->conf.beta = para[index]["beta"];
            l->conf.transB = para[index]["transB"];
            // weight and bias
            std::vector<int> weight_shape = para[index]["weight_shape"];
            l->conf.weight_shape = weight_shape;
            l->conf.bias_shape = para[index]["bias_shape"];

            int ws = vector_multi(l->conf.weight_shape);
            int bs = l->conf.bias_shape;
            float* src_weight = new float[ws];
            float* src_bias = new float[bs];
            std::string WeightFile = paras_dir + std::string(para[index]["weight"]);
            std::string BiasFile = paras_dir + std::string(para[index]["bias"]);
            
            std::ifstream i(WeightFile, std::ios::in | std::ios::binary);
            i.read((char *)src_weight, ws * sizeof(float));
            i.close();
            std::ifstream j(BiasFile, std::ios::in | std::ios::binary);
            j.read((char *)src_bias, bs * sizeof(float));
            j.close();

            l->conf.weight = copytogpu(src_weight, l->conf.weight, ws);
            l->conf.bias = copytogpu(src_bias, l->conf.bias, bs);
            delete []src_weight;
            delete []src_bias;
            // input and output
            l->conf.input_shape = data_out;
            data_out[0] = l->conf.bias_shape;
            l->conf.input = gpumalloc(l->conf.input,vector_multi(l->conf.input_shape));
            l->conf.output_shape = data_out;
            l->conf.output = gpumalloc(l->conf.output,vector_multi(l->conf.output_shape));
            paras.push_back((layer *)l);
        }else if(type == "Flatten"){
            std::vector<int> v(2,1);
            v[0] = vector_multi(data_out);
            data_out = v;
        }else {
            layer *l = new layer(name,type);
            paras.push_back((layer *)l);
        }
    }

    in.close();
    std::cout << std::endl
              << "load paras finished" << std::endl;
}

void VGG16::print_message(){
    for(auto ly:paras){        
        if(ly->info.type == "Conv"){
            convolution *l = dynamic_cast<convolution*>(ly);
            l->print_message();
        }else if(ly->info.type == "MaxPool"){
            maxpool *l = dynamic_cast<maxpool*>(ly);
            l->print_message();
        }else if(ly->info.type == "Gemm"){
            gemm *l = dynamic_cast<gemm*>(ly);
            l->print_message();
        }else {
           ly->print_message();
        }
    }
}

void VGG16::inference(float* in, float* out){
    input = copytogpu(in, input, vector_multi(input_shape));
    float * last_out = input;
    for(auto ly:paras){     
        // if(ly->info.name == "Conv_5"){
        //     break;
        // }
        if(ly->info.type == "Conv"){
            convolution *l = dynamic_cast<convolution*>(ly);
            int dim1 = l->conf.weight_shape[0];
            int dim2 = 1024;
            int dim3 = vector_multi(l->conf.weight_shape) / dim1;
            // printf("%d %d %d %d %d %d\n", dim1, dim2, dim3, l->conf.input_shape[1], l->conf.input_shape[2], l->conf.input_shape[3]);
            Conv<<<dim1, dim2, dim3 * sizeof(float)>>>(last_out, l->conf.output, l->conf.weight, l->conf.bias, l->conf.input_shape[1], l->conf.input_shape[2], l->conf.input_shape[3]);
            cudaDeviceSynchronize();
            last_out = l->conf.output;
  
        }else if(ly->info.type == "MaxPool"){
            maxpool *l = dynamic_cast<maxpool*>(ly);
            int dim1 = l->conf.input_shape[1];
            int dim2 = 1024;
            // printf("Maxpool:%d %d\n",dim1,dim2);
            // printf("Maxpool paras:%d %d\n",l->conf.output_shape[2],l->conf.output_shape[3]);
            MaxPool<<<dim1, dim2>>>(last_out, l->conf.output,l->conf.input_shape[2], l->conf.input_shape[3],l->conf.output_shape[2], l->conf.output_shape[3]);
            cudaDeviceSynchronize();
            last_out = l->conf.output;
            
        }else if(ly->info.type == "Gemm"){
            gemm *l = dynamic_cast<gemm*>(ly);
            if(ly->info.name == "Gemm_37"){
                Gemm1<<<8, 1024>>>(l->conf.weight, last_out, l->conf.bias, l->conf.output, l->conf.weight_shape[0], l->conf.weight_shape[1], l->conf.input_shape[1]);
                cudaDeviceSynchronize();
            }else{
                Gemm2<<<8, 1024>>>(l->conf.weight, last_out, l->conf.bias, l->conf.output, l->conf.weight_shape[0], l->conf.weight_shape[1], l->conf.input_shape[1]);
                cudaDeviceSynchronize();
            }
           
            last_out = l->conf.output;
        }
    }
    copyfromgpu(last_out, out, 1000);
    output = out;
}

VGG16::~VGG16(){
    cudaFree(input);
    cudaFree(output);
    for(int i = 0; i < paras.size();i++){
        delete paras[i];
    }
}

