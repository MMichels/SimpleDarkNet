import simple_darknet

dn = simple_darknet.SimpleDarknet('imagens/teste1.jpg', 'cfg/yolov3.cfg',
                                  'weigths/yolov3.weights', 'classes/yolo.txt')

dn.run_deteccao()
dn.exibir_results()
