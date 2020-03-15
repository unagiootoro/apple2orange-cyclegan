# Load the saved DCGAN model and save the Generator weight.

require "dnn"
require_relative "dcgan"

# Epoch of the model to load.
epoch = 10

cycle_gan_model = CycleGANModel.load("trained/cycle_gan_model_epoch#{epoch}.marshal")
dcgan_A = cycle_gan_model.dcgan_A
gen_A = dcgan_A.gen1
gen_B = dcgan_A.gen2
gen_A.save_params("trained/trained_generator_A_params.marshal")
gen_B.save_params("trained/trained_generator_B_params.marshal")
