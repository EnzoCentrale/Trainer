use std::fs::File;
use std::path::Path;
use candle_core::{Device, Tensor, Result};
use image::{GenericImageView,imageops::FilterType::Triangle};

fn load_image(path : &str, size: (u32,u32)) -> Result<Tensor> {
    let img= image::open(Path::new(path)).map_err(candle_core::Error::msg)?; 

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
    let tensor = Tensor::from_vec(tensor_data, (3, height as usize,width as usize), &Device::Cpu,)?; // don't forget to define the device before

    Ok(tensor)
}

fn load_image_batch(dir_path : &str ,image_paths : &[String], size: (u32,u32) ) -> Result<Tensor> {
    let mut images = Vec::new(); // here a vector is like a list 

    for path in image_paths {
        let full_path = format!("{}/{}",dir_path,path);
        let img_tensor = load_image(&full_path, size)?;
        images.push(img_tensor);
    }

    Tensor::stack(&images,0) // the first dimension will be the image number [number ,3 ,size.0 ,size.1]
}

pub fn process_image_example() -> Result<()> {

    // i'm using this for a test, it won't work to train 
    let image_paths = vec![
        "00001.jpg".to_string(),
        "00002.jpg".to_string(),
        "00003.jpg".to_string(),
    ];

    let batch = load_image_batch("Stanford_Car_Dataset/cars_train/cars_train", &image_paths, (100, 100))?;
    println!("Batch shape: {:?}", batch.shape());
    
    // Apply normalization with ImageNet mean and std
    let mean = Tensor::new(&[0.485, 0.456, 0.406], &Device::Cpu)?.reshape((3, 1, 1))?;
    let std = Tensor::new(&[0.229, 0.224, 0.225], &Device::Cpu)?.reshape((3, 1, 1))?;
    
    let normalized_batch = batch.broadcast_sub(&mean)?.broadcast_div(&std)?;
    println!("Normalized batch shape: {:?}", normalized_batch.shape());
    
    Ok(())
}
