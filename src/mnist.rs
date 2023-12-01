use std::{fs::File, io::{self, Read}};

use crate::neural_network::InOut;

pub const IMAGE_SIZE: usize = 28 * 28;

const TRAIN_IMAGES_FILE: &str = "data/train-images.idx3-ubyte";
const TRAIN_LABELS_FILE: &str = "data/train-labels.idx1-ubyte";

const IMAGE_MAGIC_NUMBER: u32 = 0x00000803;
const LABEL_MAGIC_NUMBER: u32 = 0x00000801;

pub fn read<const N: usize>() -> io::Result<(Box<[[InOut; IMAGE_SIZE]; N]>, Box<[u8; N]>)> {
    let mut file = File::open(TRAIN_IMAGES_FILE)?;

    let magic_number = read_u32(&mut file)?;
    assert_eq!(magic_number, IMAGE_MAGIC_NUMBER);

    let _number_of_images = read_u32(&mut file)?;
    let _number_of_rows = read_u32(&mut file)?;
    let _number_of_columns = read_u32(&mut file)?;

    let mut images = Box::new([[InOut::default(); IMAGE_SIZE]; N]);

    for i in 0..N {
        let mut image = [InOut::default(); IMAGE_SIZE];
        for j in 0..IMAGE_SIZE {
            image[j] = read_u8(&mut file)? as InOut / u8::MAX as InOut;
        }
        images[i] = image;
    }

    let mut file = File::open(TRAIN_LABELS_FILE)?;

    let magic_number = read_u32(&mut file)?;
    assert_eq!(magic_number, LABEL_MAGIC_NUMBER);

    let _num_labels = read_u32(&mut file)?;

    let mut labels = Box::new([u8::default(); N]);

    for i in 0..N {
        labels[i] = read_u8(&mut file)?;
    }

    Ok((images, labels))
}

fn read_u32<T: Read>(reader: &mut T) -> io::Result<u32> {
    let mut buffer = [0; 4];
    reader.read_exact(&mut buffer)?;
    Ok(u32::from_be_bytes(buffer))
}

fn read_u8<T: Read>(reader: &mut T) -> io::Result<u8> {
    let mut buffer = [0; 1];
    reader.read_exact(&mut buffer)?;
    Ok(buffer[0])
}