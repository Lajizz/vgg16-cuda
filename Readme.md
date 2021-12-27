### VGG 模型(CUDA实现)
-------
* 文件说明
  * paras.json:模型各层类型，参数。
  * vgg16_main.cu:模型主函数，负责读取数据，初始化，调用推理函数，对比正确性
  * model.hpp:模型定义，推理
  * layers.cuh:定义具体层操作
  * utils.h:一些帮助函数
  * json.hpp:c++ 读取json 库
  * Makefile:make,make clean
* 步骤
  1. 模型解析
  2. 模型读取
  3. 模型编写-cuda
  4. 测试
  5. 优化