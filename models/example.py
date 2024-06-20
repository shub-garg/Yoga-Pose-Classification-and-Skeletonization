from inceptionresnetv2 import InceptionResNetV2
from efficientnet import EfficientNet
from densenet import DenseNet
from vgg16 import VGG16
from resnet50 import ResNet50
from inceptionv3 import InceptionV3
from mobilenetv2 import MobileNetV2

vgg16_model = VGG16(input_shape=(224, 224, 3), num_classes=1000)

resnet50_model = ResNet50(input_shape=(224, 224, 3), classes=1000)

inceptionv3_model = InceptionV3(input_shape=(299, 299, 3), num_classes=1000)

mobilenetv2_model = MobileNetV2(input_shape=(224, 224, 3), num_classes=1000)

inceptionresnetv2_model = InceptionResNetV2(input_shape=(299, 299, 3), num_classes=1000)

efficientnet_model = EfficientNet(input_shape=(224, 224, 3), num_classes=1000)

densenet_model = DenseNet(input_shape=(224, 224, 3), num_classes=1000)
