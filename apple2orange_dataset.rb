require "zip"
require "dnn/image"
require "dnn/datasets/downloader"

module DNN
  module Apple2Orange
    APPLE2ORANGE_URL = "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/"
    DIR_NAME = "apple2orange"
    DIR_PATH = "#{DOWNLOADS_PATH}/downloads/#{DIR_NAME}"

    def self.downloads
      unless Dir.exist?(DIR_PATH) 
        zip_file_name = "#{DIR_NAME}.zip"
        Downloader.download("#{APPLE2ORANGE_URL}/#{zip_file_name}")
        zip_decompression(DIR_PATH)
        File.unlink("#{DOWNLOADS_PATH}/downloads/#{zip_file_name}")
      end
    end

    def self.zip_decompression(zip_path)
      Zip::File.open("#{zip_path}.zip") do |zip|
        zip.each do |entry|
          zip.extract(entry, "#{zip_path}/#{entry.name}") { true }
        end
      end
    end

    def self.load_images(dir_path, resize_size: nil)
      downloads
      imgs = []
      Dir["#{dir_path}/*.jpg"].each do |fpath|
        img = DNN::Image.read(fpath)
        img = DNN::Image.resize(img, *resize_size) if resize_size
        imgs << img
      end
      imgs
    end

    def self.load_train(resize_size: [64, 64])
      x = load_images("#{DIR_PATH}/#{DIR_NAME}/trainA", resize_size: resize_size)
      y = load_images("#{DIR_PATH}/#{DIR_NAME}/trainB", resize_size: resize_size)
      [x, y]
    end

    def self.load_test(resize_size: [64, 64])
      x = load_images("#{DIR_PATH}/#{DIR_NAME}/testA", resize_size: resize_size)
      y = load_images("#{DIR_PATH}/#{DIR_NAME}/testB", resize_size: resize_size)
      [x, y]
    end
  end
end
