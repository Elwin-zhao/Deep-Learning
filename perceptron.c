#include <stdio.h>
#include <stdlib.h>

typedef struct preceptron {
    float bias;      //偏置项
    float weight[2]; //权重
    float rate;      //学习率
}preceptron;
preceptron *Init(void) {
    int i = 0;      
    preceptron *p = (preceptron *)malloc(sizeof(preceptron));
    p->bias = 0.0;  //初始化偏置项为0
    for(i = 0; i < 2; ++i) {
        p->weight[i] = 0.0;   //初始化权重为0
    }
    p->rate = 0.1;  //学习率为0.1
    return p;
}
int activator(float x) {  //定义激活函数为阶跃函数
    if(x > 0) {
        return 1;
    }
    else {
        return 0;
    }
}
float Output(float inputvec[],preceptron *p) {  //计算每次的输出
    float sum = 0;
    int i;

    for(i = 0; i < 2; ++i) {
        sum = sum + inputvec[i] * p->weight[i]; //输入向量和权重求内积
    }
    sum = sum + p->bias;  //加上偏置项
    return activator(sum);  //经过激活函数后返回0或1
}
void Update_weight (float input_vec[], float output, float label, preceptron *p) { //根据感知器算法更新权重
    float delta = label - output;
    int i;
    for(i = 0; i < 2; ++i) {
        input_vec[i] = input_vec[i] * delta * p->rate;
    }
    for(i = 0; i < 2; ++i) {
        p->weight[i] = p->weight[i] + input_vec[i];
    }
    p->bias = p->bias + p->rate * delta;
}
void One_Iteration (float input_vecs[][2], float labels[], preceptron *p) { //一轮迭代
    int i;
    float output,inputvec[2];

    for(i = 0; i < 4; ++i) {
        inputvec[0] = input_vecs[i][0];
        inputvec[1] = input_vecs[i][1];
        output = Output(inputvec, p);
        Update_weight(inputvec, output, labels[i], p);
    }
}
void Preceptron (float input_vecs[][2], float labels[], preceptron *p, int Itertation) {
    int i;
    for(i = 0; i < Itertation; ++i) {
        One_Iteration(input_vecs, labels, p);
    }
}
int main () {
    float input_vecs[4][2] = {{1,1},{0,0},{1,0},{0,1}}; //感知器实现与门
    float labels[4] = {1,0,0,0};  //对应结果
    float inputvec[2];
    preceptron *p = Init();
    Preceptron(input_vecs, labels, p, 10);
    printf("weights:[%.1f, %.1f]\n",p->weight[0],p->weight[1]);
    printf("bias   :%.6f\n",p->bias);

    //测试
    inputvec[0] = inputvec[1] = 1;
    printf("1 and 1 = %.0f\n",Output(inputvec,p));
    inputvec[0] = inputvec[1] = 0;
    printf("0 and 0 = %.0f\n",Output(inputvec,p));
    inputvec[0] = 1;
    inputvec[1] = 0;
    printf("1 and 0 = %.0f\n",Output(inputvec,p));
    inputvec[0] = 0;
    inputvec[1] = 1;
    printf("0 and 1 = %.0f\n",Output(inputvec,p));
    return 0;
}
