#include "/home/jachinshen/Apps/caffe/include/caffe/caffe.hpp"
#include <string>
#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace caffe;
using namespace cv;
using namespace std;

unsigned int get_blob_index(boost::shared_ptr< Net<float> > & net, char *query_blob_name)
{
    std::string str_query(query_blob_name);    
    vector< string > const & blob_names = net->blob_names();
    for( unsigned int i = 0; i != blob_names.size(); ++i ) 
    { 
        if( str_query == blob_names[i] ) 
        { 
            return i;
        } 
    }
}

void caffe_forward(boost::shared_ptr< Net<float> > & net, float *data_ptr)
{
    char query_blob_name[10] = "data"; /* data, conv1, pool1, norm1, fc6, prob, etc */
    unsigned int blob_id = get_blob_index(net, query_blob_name);

    cout<<"data blob id:"<<blob_id<<endl;
    Blob<float>* input_blobs = net->input_blobs()[blob_id];
    switch (Caffe::mode())
    {
        case Caffe::CPU:
            memcpy(input_blobs->mutable_cpu_data(), data_ptr,
                    sizeof(float) * input_blobs->count());
            break;
            /*case Caffe::GPU:
              cudaMemcpy(input_blobs->mutable_gpu_data(), data_ptr,
              sizeof(float) * input_blobs->count(), cudaMemcpyHostToDevice);
              break;\
              */
        default: break;
    } 
    //net->ForwardPrefilled();
    net->Forward();
}

//! Note: data_ptr指向已经处理好（去均值的，符合网络输入图像的长宽和Batch Size）的数据
int main()
{
    char proto[100] = "/home/jachinshen/Apps/caffe/examples/cpp_minst/lenet_deploy.prototxt"; /* 加载CaffeNet的配置 */
    Phase phase = TEST; /* or TRAIN */
    Caffe::set_mode(Caffe::CPU);
    boost::shared_ptr< Net<float> > net(new caffe::Net<float>(proto, phase));
    char model[100] = "/home/jachinshen/Apps/caffe/examples/cpp_minst/lenet_iter_10000.caffemodel";    
    net->CopyTrainedLayersFrom(model);

    Mat img_raw = imread("/home/jachinshen/Apps/caffe/examples/cpp_minst/_00175.png", 1); 
    Mat img;
    cvtColor(img_raw, img, CV_BGR2GRAY);
    Mat data(28, 28, CV_32FC1);
    resize(img, img, Size(28, 28));
    threshold(img, img, 128, 1, THRESH_BINARY_INV);
    imshow("img", img);
    waitKey(0);
    cout<< img<<endl;
    //normalize(data, data, 0.0, 1.0, NORM_MINMAX);

    float data_ptr[784];
    uchar* p_data;
    for( int i=0; i<28; i++)
    {
        p_data = img.ptr< uchar>(i);
        for(int j=0; j<28; j++)
        {
        	data_ptr[28*i+j]=(float)*(p_data+j);
        }
    }
    //for(int i=0;i<784;i++)
    //    cout<<*(data_ptr+i);
    caffe_forward(net, data_ptr);

    char query_blob_name[10] = "loss"; /* data, conv1, pool1, norm1, fc6, prob, etc */
    unsigned int blob_id = get_blob_index(net, query_blob_name);

    boost::shared_ptr<Blob<float> > blob = net->blobs()[blob_id];
    unsigned int num_data = blob->count(); /* NCHW=10x96x55x55 */
    std::cout<<"data number:"<<num_data<<std::endl;
    const float *blob_ptr = (const float *) blob->cpu_data();
    /*for( int i=0; i<28; i++)
    {
        for(int j=0; j<28; j++)
        {
        std::cout<<*(blob_ptr+i*28+j)<<" ";
        }
        std::cout<<std::endl;
    }*/
    for( int i=0; i<num_data; i++)
    {
       cout<<*(blob_ptr+i)<<" ";
    }
    return 0;
}