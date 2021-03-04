
#include "stdafx.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include<random>

#include "tiny_dnn/tiny_dnn.h"
using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;
using namespace std;



void sample1_convnet(const string& data_dir = "../data");
int main(int argc, char** argv) {
    try {
        if (argc == 2) {
            sample1_convnet(argv[1]);
        }
        else {
            sample1_convnet();
        }
    }
    catch (const nn_error& e) {
        std::cout << e.what() << std::endl;
    }
    
    
    return 0;
}


//！ learning convolutional neural networks (LeNet-5 like architecture)
void sample1_convnet(const std::string& data_dir) {
    //! construct LeNet-5 architecture
    tiny_dnn::network<sequential> nn;
    tiny_dnn::adagrad optimizer;

    

    using conv = tiny_dnn::layers::conv;
    using ave_pool = tiny_dnn::layers::ave_pool;
    using max_pool = tiny_dnn::layers::max_pool;
    using dropout = tiny_dnn::layers::dropout;
    using fc = tiny_dnn::layers::fc;
    using tanh = tiny_dnn::activation::tanh;

   
    //! four convolution layers, two max pooling layers, two fully connected layers
    nn << conv(32, 32, 5, 1, 6) //! 32x32 in, 5x5 kernel, 1-6 fmaps conv 
        << tanh(28, 28, 6)
        <<conv(28,28,5,6,10)
        <<tanh(24,24,10)
        << max_pool(24, 24, 10, 2) //! 24x24 in, 10 fmaps, 2x2 subsampling
        << tanh(12, 12, 10)
        << conv(12, 12, 5, 10, 16)
        << tanh(8, 8, 16) << max_pool(8, 8, 16, 2) << tanh(4, 4, 16)
        << conv(4, 4, 4, 16, 120) << tanh(1, 1, 120) << fc(120, 64) <<fc(64,10)<< tanh(10);

    std::cout << "load models..." << std::endl;

    //! load MNIST dataset
    std::vector<label_t> train_labels, test_labels;
    std::vector<vec_t> train_images, test_images;

    std::string train_labels_path = "D:\\datasets\\data\\train-labels.idx1-ubyte";
    std::string train_images_path = "D:\\datasets\\data\\train-images.idx3-ubyte";
    std::string test_labels_path = "D:\\datasets\\data\\t10k-labels.idx1-ubyte";
    std::string test_images_path = "D:\\datasets\\data\\t10k-images.idx3-ubyte";

    parse_mnist_labels(train_labels_path, &train_labels);
    parse_mnist_images(train_images_path, &train_images, -1.0, 1.0, 2, 2);
    parse_mnist_labels(test_labels_path, &test_labels);
    parse_mnist_images(test_images_path, &test_images, -1.0, 1.0, 2, 2);

    std::cout << "start learning" << std::endl;

    progress_display disp(train_images.size());
    timer t;
    int minibatch_size = 10;

    optimizer.alpha *= std::sqrt(minibatch_size);

    //! create callback
    auto on_enumerate_epoch = [&]() {
        std::cout << t.elapsed() << "s elapsed." << std::endl;

        tiny_dnn::result res = nn.test(test_images, test_labels);

        std::cout << res.num_success << "/" << res.num_total << std::endl;

        disp.restart(train_images.size());
        t.restart();
    };

    auto on_enumerate_minibatch = [&]() { disp += minibatch_size; };

    //! training
    nn.train<mse>(optimizer, train_images, train_labels, minibatch_size, 5,
        on_enumerate_minibatch, on_enumerate_epoch);

    std::cout << "end training." << std::endl;

    //! test and show results
    nn.test(test_images, test_labels).print_detail(std::cout);

    //! save networks
    std::ofstream ofs("LeNet-weights");
    ofs << nn;
}



