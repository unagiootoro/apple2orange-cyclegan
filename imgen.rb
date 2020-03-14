require "dnn"
require "dnn/image"
require "numo/linalg/autoloader"
require_relative "apple2orange_dataset"
require_relative "dcgan"

def load_dataset
  x, y = DNN::Apple2Orange.load_test
  x = Numo::SFloat.cast(x)
  y = Numo::SFloat.cast(y)
  x = (x / 127.5) - 1
  y = (y / 127.5) - 1
  [x, y]
end

x_in, x_out = load_dataset
batch_size = x_out.shape[0]

# Load the Generator model.
gen_A = Generator.new([64, 64, 3], 64)
gen_A.predict1(Numo::SFloat.zeros(64, 64, 3))
gen_A.load_params("trained/trained_generator_A_params.marshal")
gen_B = Generator.new([64, 64, 3], 64)
gen_B.predict1(Numo::SFloat.zeros(64, 64, 3))
gen_B.load_params("trained/trained_generator_B_params.marshal")

batch_size.times do |i|
  # Save the input image.
  input = x_in[i, false]
  img = Numo::UInt8.cast(((input + 1) * 127.5).round)
  DNN::Image.write("img/img_#{i}_input.jpg", img)

  # Save the output image.
  out = gen_A.predict1(x_in[i, false])
  img = Numo::UInt8.cast(((out + 1) * 127.5).round)
  DNN::Image.write("img/img_#{i}_output.jpg", img)
end

batch_size.times do |i|
  # Save the input image.
  input = x_out[i, false]
  img = Numo::UInt8.cast(((input + 1) * 127.5).round)
  DNN::Image.write("img/img2_#{i}_input.jpg", img)

  # Save the output image.
  out = gen_B.predict1(x_out[i, false])
  img = Numo::UInt8.cast(((out + 1) * 127.5).round)
  DNN::Image.write("img/img2_#{i}_output.jpg", img)
end
