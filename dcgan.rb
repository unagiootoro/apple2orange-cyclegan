include DNN::Models
include DNN::Layers

class Generator < Model
  def initialize(input_shape, base_num_filters)
    super()
    @input_shape = input_shape
    @cv1 = Conv2D.new(base_num_filters, 4, padding: true)
    @cv2 = Conv2D.new(base_num_filters, 4, strides: 2, padding: true)
    @cv3 = Conv2D.new(base_num_filters * 2, 4, padding: true)
    @cv4 = Conv2D.new(base_num_filters * 2, 4, strides: 2, padding: true)
    @cv5 = Conv2D.new(base_num_filters * 2, 4, padding: true)
    @cv6 = Conv2D.new(base_num_filters, 4, padding: true)
    @cv7 = Conv2D.new(base_num_filters, 4, padding: true)
    @cv8 = Conv2D.new(3, 4, padding: true)
    @cvt1 = Conv2DTranspose.new(base_num_filters * 2, 4, strides: 2, padding: true)
    @cvt2 = Conv2DTranspose.new(base_num_filters, 4, strides: 2, padding: true)
    @bn1 = BatchNormalization.new
    @bn2 = BatchNormalization.new
    @bn3 = BatchNormalization.new
    @bn4 = BatchNormalization.new
    @bn5 = BatchNormalization.new
    @bn6 = BatchNormalization.new
    @bn7 = BatchNormalization.new
    @bn8 = BatchNormalization.new
  end

  def forward(x)
    input = InputLayer.new(@input_shape).(x)
    x = @cv1.(input)
    x = @bn1.(x)
    h1 = LeakyReLU.(x, 0.2)

    x = @cv2.(h1)
    x = @bn2.(x)
    x = LeakyReLU.(x, 0.2)

    x = @cv3.(x)
    x = @bn3.(x)
    h2 = LeakyReLU.(x, 0.2)

    x = @cv4.(h2)
    x = @bn4.(x)
    x = LeakyReLU.(x, 0.2)

    x = @cv5.(x)
    x = @bn5.(x)
    x = LeakyReLU.(x, 0.2)

    x = @cvt1.(x)
    x = @bn6.(x)
    x = LeakyReLU.(x, 0.2)
    x = Concatenate.(x, h2, axis: 3)

    x = @cv6.(x)
    x = @bn7.(x)
    x = LeakyReLU.(x, 0.2)

    x = @cvt2.(x)
    x = @bn8.(x)
    x = LeakyReLU.(x, 0.2)
    x = Concatenate.(x, h1, axis: 3)

    x = @cv7.(x)
    x = LeakyReLU.(x, 0.2)

    x = @cv8.(x)
    x = Tanh.(x)
    x
  end
end

class Discriminator < Model
  def initialize(input_shape, base_num_filters)
    super()
    @input_shape = input_shape
    @cv1 = Conv2D.new(base_num_filters, 4, padding: true)
    @cv2 = Conv2D.new(base_num_filters, 4, strides: 2, padding: true)
    @cv3 = Conv2D.new(base_num_filters * 2, 4, padding: true)
    @cv4 = Conv2D.new(base_num_filters * 2, 4, strides: 2, padding: true)
    @d1 = Dense.new(1024)
    @d2 = Dense.new(1)
    @bn1 = BatchNormalization.new
    @bn2 = BatchNormalization.new
    @bn3 = BatchNormalization.new
    @bn4 = BatchNormalization.new
  end

  def forward(x)
    x = InputLayer.new(@input_shape).(x)
    x = @cv1.(x)
    x = @bn1.(x)
    x = LeakyReLU.(x, 0.2)

    x = @cv2.(x)
    x = @bn2.(x)
    x = LeakyReLU.(x, 0.2)

    x = @cv3.(x)
    x = @bn3.(x)
    x = LeakyReLU.(x, 0.2)

    x = @cv4.(x)
    x = @bn4.(x)
    x = LeakyReLU.(x, 0.2)

    x = Flatten.(x)
    x = @d1.(x)
    x = LeakyReLU.(x, 0.2)

    x = @d2.(x)
    x
  end

  def enable_training
    trainable_layers.each do |layer|
      layer.trainable = true
    end
  end

  def disable_training
    trainable_layers.each do |layer|
      layer.trainable = false
    end
  end
end

class DCGAN < Model
  attr_reader :gen1
  attr_reader :gen2
  attr_reader :dis

  def initialize(gen1, gen2, dis)
    super()
    @gen1 = gen1
    @gen2 = gen2
    @dis = dis
  end

  def forward(input)
    images = @gen1.(input)
    @dis.disable_training
    out = @dis.(images)
    cycle_image = @gen2.(images)
    [cycle_image, out]
  end
end

class CycleGANModel < Model
  attr_accessor :dcgan_A
  attr_accessor :dcgan_B

  def initialize(dcgan_A, dcgan_B)
    super()
    @dcgan_A = dcgan_A
    @dcgan_B = dcgan_B
  end
end
