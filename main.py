import simple_darknet

dn = simple_darknet.SimpleDarknet('arquivos/dog.jpg', 'arquivos/yolov3.cfg',
                                  'arquivos/yolov3.weights', 'arquivos/yolo.txt')

dn.run_deteccao()
dn.exibir_results()
