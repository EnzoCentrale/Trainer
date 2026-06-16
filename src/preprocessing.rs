use std::fs::File;
use std::path::Path;
use candle_core::Tensor;
use matfile::MatFile;
use image::{DynamicImage, GenericImageView, Rgba, imageops::FilterType::Triangle};

pub fn load_image(path : &str, size: (u32,u32)) -> Result<Tensor> {
    let img = image::open(Path::new(path))?;

    //Resizing image 
    let img = img.resize_exact(size.0,size.1,image::imageops::FilterType::Triangle);

    //Converting to RGB Tensor
    let (width,height) = img.dimensions();
    let mut tensor_data= Vec::with_capacity((width * height * 3) as usize); // Basically tensors with 3 layers ( blue red green )

    // now we gotta extract the RGB from the image 
    for pixel in img.pixels() {
        let rgb = pixel.2 ;
        tensor_data.push(rgb[0] as f32 / 255.0);
        tensor_data.push(rgb[1] as f32 / 255.0);
        tensor_data.push(rgb[2] as f32 / 255.0);
    }

    // create Tensor with shape [channels, height, width]
    let tensor = Tensor::from_vec(tensor_data, (3, height as usize,width as usize), &Device::Cpu,); // don't forget to define the device before

    Ok(tensor)
}
