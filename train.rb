require "dnn"
require "numo/linalg/autoloader"
require_relative "apple2orange_dataset"
require_relative "dcgan"

include DNN::Optimizers
include DNN::Losses

INITIAL_EPOCH = 1
EPOCHS = 10
BATCH_SIZE = 32

def load_dataset
  x, y = DNN::Apple2Orange.load_train
  x = Numo::SFloat.cast(x)
  y = Numo::SFloat.cast(y)
  x = (x / 127.5) - 1
  y = (y / 127.5) - 1
  [x, y]
end

class DNN::CycleGANIterator < DNN::Iterator
  def initialize(x_datas, y_datas, random: true, last_round_down: false)
    @x_datas = x_datas
    @y_datas = y_datas
    @random = random
    @last_round_down = last_round_down
    num_datas1 = x_datas.is_a?(Array) ? x_datas[0].shape[0] : x_datas.shape[0]
    num_datas2 = y_datas.is_a?(Array) ? y_datas[0].shape[0] : y_datas.shape[0]
    if num_datas1 < num_datas2
      @num_datas = num_datas1
    else
      @num_datas = num_datas2
    end
    reset
  end

  def next_batch(batch_size)
    raise DNN::DNNError, "This iterator has not next batch. Please call reset." unless has_next?
    if @indexes1.length <= batch_size
      batch_indexes1 = @indexes1
      batch_indexes2 = @indexes2
      @has_next = false
    else
      batch_indexes1 = @indexes1.shift(batch_size)
      batch_indexes2 = @indexes2.shift(batch_size)
    end
    x_batch, _ = get_batch(batch_indexes1)
    _, y_batch = get_batch(batch_indexes2)
    [x_batch, y_batch]
  end

  def reset
    @has_next = true
    @indexes1 = @num_datas.times.to_a
    @indexes2 = @num_datas.times.to_a
    if @random
      @indexes1.shuffle!
      @indexes2.shuffle!
    end
  end
end

if INITIAL_EPOCH == 1
  gen_A = Generator.new([64, 64, 3], 64)
  gen_B = Generator.new([64, 64, 3], 64)
  dis_A = Discriminator.new([64, 64, 3], 64)
  dis_B = Discriminator.new([64, 64, 3], 64)
  dcgan_A = DCGAN.new(gen_A, gen_B, dis_A)
  dcgan_B = DCGAN.new(gen_B, gen_A, dis_B)
  dis_A.setup(Adam.new(alpha: 0.00001, beta1: 0.1), SigmoidCrossEntropy.new)
  dis_B.setup(Adam.new(alpha: 0.00001, beta1: 0.1), SigmoidCrossEntropy.new)
  dcgan_A.setup(Adam.new(alpha: 0.0001, beta1: 0.5),
                [MeanAbsoluteError.new, SigmoidCrossEntropy.new], loss_weights: [10, 1])
  dcgan_B.setup(Adam.new(alpha: 0.0001, beta1: 0.5),
                [MeanAbsoluteError.new, SigmoidCrossEntropy.new], loss_weights: [10, 1])
  cycle_gan_model = CycleGANModel.new(dcgan_A, dcgan_B)
  # TODO: This setup is dummy. I will fix later.
  cycle_gan_model.setup(SGD.new, MeanAbsoluteError.new)
else
  cycle_gan_model = CycleGANModel.load("trained/cycle_gan_model_epoch#{INITIAL_EPOCH - 1}.marshal")
  dcgan_A = cycle_gan_model.dcgan_A
  dcgan_B = cycle_gan_model.dcgan_B
  gen_A = dcgan_A.gen1
  gen_B = dcgan_A.gen2
  dis_A = dcgan_A.dis
  dis_B = dcgan_B.dis
end

x, y = load_dataset

iter1 = DNN::CycleGANIterator.new(x, y)
iter2 = DNN::CycleGANIterator.new(x, y)
iter3 = DNN::CycleGANIterator.new(x, y)
iter4 = DNN::CycleGANIterator.new(x, y)
num_batchs = iter1.num_datas / BATCH_SIZE
real = Numo::SFloat.ones(BATCH_SIZE, 1)
fake = Numo::SFloat.zeros(BATCH_SIZE, 1)

(INITIAL_EPOCH..EPOCHS).each do |epoch|
  num_batchs.times do |index|
    x_batch, y_batch = iter1.next_batch(BATCH_SIZE)
    x_batch2, y_batch2 = iter2.next_batch(BATCH_SIZE)
    x_batch3, y_batch3 = iter3.next_batch(BATCH_SIZE)
    x_batch4, y_batch4 = iter4.next_batch(BATCH_SIZE)

    images_A = gen_A.predict(x_batch)
    dis_A.enable_training
    dis_loss = dis_A.train_on_batch(y_batch, real)
    dis_loss += dis_A.train_on_batch(images_A, fake)

    dcgan_loss = dcgan_A.train_on_batch(x_batch2, [x_batch2, real])

    puts "A epoch: #{epoch}, index: #{index}, dis_loss: #{dis_loss}, dcgan_loss: #{dcgan_loss}"

    images_B = gen_B.predict(y_batch3)
    dis_B.enable_training
    dis_loss = dis_B.train_on_batch(x_batch3, real)
    dis_loss += dis_B.train_on_batch(images_B, fake)

    dcgan_loss = dcgan_B.train_on_batch(y_batch4, [y_batch4, real])

    puts "B epoch: #{epoch}, index: #{index}, dis_loss: #{dis_loss}, dcgan_loss: #{dcgan_loss}"
  end
  if epoch % 5 == 0
    cycle_gan_model.save("trained/cycle_gan_model_epoch#{epoch}.marshal")
  end
  iter1.reset
  iter2.reset
  iter3.reset
  iter4.reset
end
