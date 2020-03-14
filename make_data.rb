# Load the saved DCGAN model and save the Generator weight.

require "dnn"
require_relative "dcgan"

# Epoch of the model to load.
epoch = 10

dcgan_A = DCGAN.load("trained/dcgan_A_model_epoch#{epoch}.marshal")
gen_A = dcgan_A.gen1
gen_B = dcgan_A.gen2
gen_A.save_params("trained/trained_generator_A_params.marshal")
gen_B.save_params("trained/trained_generator_B_params.marshal")
