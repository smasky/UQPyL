#include "svm.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py=pybind11;

// svm_type kernel_type degree gamma coef0 C nu p eps
struct Parameter
{
    int svm_type;
    int kernel_type;
    int degree;
    int max_Iter;
    float gamma;
    float coef0;
    float C;
    float nu;
    float p;
    float eps;
};

struct SvmHandle
{
    svm_model* model = nullptr;
    svm_node** trainNodes = nullptr;
    int nSamples = 0;

    ~SvmHandle()
    {
        if (model != nullptr) {
            svm_free_and_destroy_model(&model);
        }
        if (trainNodes != nullptr) {
            for (int row = 0; row < nSamples; ++row) {
                delete[] trainNodes[row];
            }
            delete[] trainNodes;
        }
    }
};

void print_null(const char *){
}

std::unique_ptr<SvmHandle> fit(py::array_t<double, py::array::c_style> trainX,
            py::array_t<double, py::array::c_style> trainY, Parameter& par)
{   
    svm_set_print_string_function(print_null);
    py::buffer_info X=trainX.request();
    py::buffer_info Y=trainY.request();
    int n_samples=X.shape[0];
    int n_features=X.shape[1];

    int l=n_samples;
    double *Ptr_y=static_cast<double *>(Y.ptr);
    
    double *Ptr_x=static_cast<double *>(X.ptr);
    auto handle = std::make_unique<SvmHandle>();
    handle->nSamples = n_samples;
    handle->trainNodes = new svm_node*[n_samples];
    
    for (int row=0; row<n_samples;row++){
        handle->trainNodes[row]=new svm_node[n_features+1];
        for (int col=0; col<n_features;col++){
            handle->trainNodes[row][col].index=col + 1;
            handle->trainNodes[row][col].value=Ptr_x[n_features*row+col];
        }
        handle->trainNodes[row][n_features].index=-1;
    }

    svm_problem* Pro= new svm_problem;
    Pro->l=l;Pro->x=handle->trainNodes;Pro->y=Ptr_y;
    //***************************************
    svm_parameter* Para=new svm_parameter;
    Para->svm_type=par.svm_type;
    Para->kernel_type=par.kernel_type;
    Para->degree=par.degree;
    Para->gamma=par.gamma;
    Para->coef0=par.coef0;
    Para->C=par.C;
    Para->nu=par.nu;
    Para->p=par.p;
    Para->eps=par.eps;
    Para->max_Iter=par.max_Iter;
    Para->cache_size=100;
    Para->nr_weight=0;
    Para->weight_label=nullptr;
    Para->weight=nullptr;
    Para->shrinking=1;
    Para->probability=0;

    const char* error_msg = svm_check_parameter(Pro, Para);
    if(error_msg != nullptr){
        delete Pro;
        delete Para;
        throw std::runtime_error(error_msg);
    }
    //**************************************
    handle->model = svm_train(Pro, Para);
    
    delete Pro;
    delete Para;

    return handle;
}

double predict(const SvmHandle& handle, py::array_t<double, py::array::c_style> testX){

    py::buffer_info X=testX.request();
    double *x_Ptr=static_cast<double *>(X.ptr);
    int n_features=X.shape[0];
    svm_node* x=new svm_node[n_features+1];
    for (int i=0;i<n_features;i++){
        x[i].index=i + 1;
        x[i].value=x_Ptr[i];
    }
    x[n_features].index=-1;

    double re=svm_predict(handle.model, x);
    delete[] x;
    return re;
}

PYBIND11_MODULE(libsvm_interface, m)
{
    m.doc() = "Svm Core Code";
    m.def("svm_fit", &fit, "svm_fit");
    m.def("svm_predict", &predict, "svm_predict");
    py::class_<Parameter>(m, "Parameter")
        .def(py::init<int, int, int, int, float, float, float, float, float ,float>())
        .def_readwrite("svm_type", &Parameter::svm_type)
        .def_readwrite("kernel_type", &Parameter::kernel_type)
        .def_readwrite("degree", &Parameter::degree)
        .def_readwrite("maxIter",&Parameter::max_Iter)
        .def_readwrite("gamma", &Parameter::gamma)
        .def_readwrite("coef0", &Parameter::coef0)
        .def_readwrite("C", &Parameter::C)
        .def_readwrite("nu", &Parameter::nu)
        .def_readwrite("p", &Parameter::p)
        .def_readwrite("eps", &Parameter::eps);

    py::class_<SvmHandle>(m, "svm_model");

}       
