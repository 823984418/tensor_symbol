use std::fmt::{Debug, Formatter};
use std::fs::File;
use std::io;
use std::io::{BufReader, BufWriter, ErrorKind, Read, Write};
use std::ops::Deref;
use std::path::Path;

use byteorder::{ReadBytesExt, WriteBytesExt};

pub trait IdxDataType {
    type Type: Copy;
    const TYPE: u8;
    fn get_vec(data: IdxData) -> Result<Vec<Self::Type>, IdxData>;
    fn get_slice(data: &IdxData) -> Option<&[Self::Type]>;
    fn read<R: Read>(input: &mut R) -> Result<Self::Type, io::Error>;
    fn write<W: Write>(output: &mut W, v: Self::Type) -> Result<(), io::Error>;
    fn read_all<R: Read>(input: &mut R, size: usize) -> Result<Vec<Self::Type>, io::Error> {
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(Self::read(input)?);
        }
        Ok(data)
    }
    fn write_all<W: Write>(output: &mut W, v: &[Self::Type]) -> Result<(), io::Error> {
        for &i in v {
            Self::write(output, i)?;
        }
        Ok(())
    }
}

pub struct IdxDataWarp<T>(T);

impl IdxDataType for IdxDataWarp<u8> {
    type Type = u8;
    const TYPE: u8 = 0x08;

    fn get_vec(data: IdxData) -> Result<Vec<Self::Type>, IdxData> {
        match data {
            IdxData::U8 { data } => Ok(data),
            _ => Err(data),
        }
    }

    fn get_slice(data: &IdxData) -> Option<&[Self::Type]> {
        match data {
            IdxData::U8 { data } => Some(data),
            _ => None,
        }
    }
    fn read<R: Read>(input: &mut R) -> Result<Self::Type, io::Error> {
        input.read_u8()
    }

    fn write<W: Write>(output: &mut W, v: Self::Type) -> Result<(), io::Error> {
        output.write_u8(v)
    }
}
impl IdxDataType for IdxDataWarp<i8> {
    type Type = i8;
    const TYPE: u8 = 0x09;

    fn get_vec(data: IdxData) -> Result<Vec<Self::Type>, IdxData> {
        match data {
            IdxData::I8 { data } => Ok(data),
            _ => Err(data),
        }
    }

    fn get_slice(data: &IdxData) -> Option<&[Self::Type]> {
        match data {
            IdxData::I8 { data } => Some(data),
            _ => None,
        }
    }
    fn read<R: Read>(input: &mut R) -> Result<Self::Type, io::Error> {
        input.read_i8()
    }
    fn write<W: Write>(output: &mut W, v: Self::Type) -> Result<(), io::Error> {
        output.write_i8(v)
    }
}
impl IdxDataType for IdxDataWarp<i16> {
    type Type = i16;
    const TYPE: u8 = 0x0B;

    fn get_vec(data: IdxData) -> Result<Vec<Self::Type>, IdxData> {
        match data {
            IdxData::I16 { data } => Ok(data),
            _ => Err(data),
        }
    }

    fn get_slice(data: &IdxData) -> Option<&[Self::Type]> {
        match data {
            IdxData::I16 { data } => Some(data),
            _ => None,
        }
    }
    fn read<R: Read>(input: &mut R) -> Result<Self::Type, io::Error> {
        input.read_i16::<byteorder::BE>()
    }
    fn write<W: Write>(output: &mut W, v: Self::Type) -> Result<(), io::Error> {
        output.write_i16::<byteorder::BE>(v)
    }
}
impl IdxDataType for IdxDataWarp<i32> {
    type Type = i32;
    const TYPE: u8 = 0x0C;

    fn get_vec(data: IdxData) -> Result<Vec<Self::Type>, IdxData> {
        match data {
            IdxData::I32 { data } => Ok(data),
            _ => Err(data),
        }
    }

    fn get_slice(data: &IdxData) -> Option<&[Self::Type]> {
        match data {
            IdxData::I32 { data } => Some(data),
            _ => None,
        }
    }
    fn read<R: Read>(input: &mut R) -> Result<Self::Type, io::Error> {
        input.read_i32::<byteorder::BE>()
    }
    fn write<W: Write>(output: &mut W, v: Self::Type) -> Result<(), io::Error> {
        output.write_i32::<byteorder::BE>(v)
    }
}
impl IdxDataType for IdxDataWarp<f32> {
    type Type = f32;
    const TYPE: u8 = 0x0D;

    fn get_vec(data: IdxData) -> Result<Vec<Self::Type>, IdxData> {
        match data {
            IdxData::F32 { data } => Ok(data),
            _ => Err(data),
        }
    }

    fn get_slice(data: &IdxData) -> Option<&[Self::Type]> {
        match data {
            IdxData::F32 { data } => Some(data),
            _ => None,
        }
    }
    fn read<R: Read>(input: &mut R) -> Result<Self::Type, io::Error> {
        input.read_f32::<byteorder::BE>()
    }
    fn write<W: Write>(output: &mut W, v: Self::Type) -> Result<(), io::Error> {
        output.write_f32::<byteorder::BE>(v)
    }
}
impl IdxDataType for IdxDataWarp<f64> {
    type Type = f64;
    const TYPE: u8 = 0x0E;

    fn get_vec(data: IdxData) -> Result<Vec<Self::Type>, IdxData> {
        match data {
            IdxData::F64 { data } => Ok(data),
            _ => Err(data),
        }
    }

    fn get_slice(data: &IdxData) -> Option<&[Self::Type]> {
        match data {
            IdxData::F64 { data } => Some(data),
            _ => None,
        }
    }
    fn read<R: Read>(input: &mut R) -> Result<Self::Type, io::Error> {
        input.read_f64::<byteorder::BE>()
    }
    fn write<W: Write>(output: &mut W, v: Self::Type) -> Result<(), io::Error> {
        output.write_f64::<byteorder::BE>(v)
    }
}

pub enum IdxData {
    U8 { data: Vec<u8> },
    I8 { data: Vec<i8> },
    I16 { data: Vec<i16> },
    I32 { data: Vec<i32> },
    F32 { data: Vec<f32> },
    F64 { data: Vec<f64> },
}

impl IdxData {
    pub fn get_type(&self) -> u8 {
        match self {
            IdxData::U8 { .. } => IdxDataWarp::<u8>::TYPE,
            IdxData::I8 { .. } => IdxDataWarp::<i8>::TYPE,
            IdxData::I16 { .. } => IdxDataWarp::<i16>::TYPE,
            IdxData::I32 { .. } => IdxDataWarp::<i32>::TYPE,
            IdxData::F32 { .. } => IdxDataWarp::<f32>::TYPE,
            IdxData::F64 { .. } => IdxDataWarp::<f64>::TYPE,
        }
    }
    pub fn data_len(&self) -> usize {
        match self {
            IdxData::U8 { data } => data.len(),
            IdxData::I8 { data } => data.len(),
            IdxData::I16 { data } => data.len(),
            IdxData::I32 { data } => data.len(),
            IdxData::F32 { data } => data.len(),
            IdxData::F64 { data } => data.len(),
        }
    }

    pub fn get_vec<T>(self) -> Result<Vec<T>, Self>
    where
        IdxDataWarp<T>: IdxDataType<Type = T>,
    {
        IdxDataWarp::<T>::get_vec(self)
    }

    pub fn get_slice<T>(&self) -> Option<&[T]>
    where
        IdxDataWarp<T>: IdxDataType<Type = T>,
    {
        IdxDataWarp::<T>::get_slice(self)
    }

    pub fn read<R: Read>(idx_type: u8, input: &mut R, size: usize) -> Result<Self, io::Error> {
        match idx_type {
            IdxDataWarp::<u8>::TYPE => Ok(IdxData::U8 {
                data: IdxDataWarp::<u8>::read_all(input, size)?,
            }),
            IdxDataWarp::<i8>::TYPE => Ok(IdxData::I8 {
                data: IdxDataWarp::<i8>::read_all(input, size)?,
            }),
            IdxDataWarp::<i16>::TYPE => Ok(IdxData::I16 {
                data: IdxDataWarp::<i16>::read_all(input, size)?,
            }),
            IdxDataWarp::<i32>::TYPE => Ok(IdxData::I32 {
                data: IdxDataWarp::<i32>::read_all(input, size)?,
            }),
            IdxDataWarp::<f32>::TYPE => Ok(IdxData::F32 {
                data: IdxDataWarp::<f32>::read_all(input, size)?,
            }),
            IdxDataWarp::<f64>::TYPE => Ok(IdxData::F64 {
                data: IdxDataWarp::<f64>::read_all(input, size)?,
            }),
            _ => Err(io::Error::new(ErrorKind::InvalidData, "type")),
        }
    }

    pub fn write<W: Write>(&self, output: &mut W) -> Result<(), io::Error> {
        match self {
            IdxData::U8 { data } => IdxDataWarp::<u8>::write_all(output, data),
            IdxData::I8 { data } => IdxDataWarp::<i8>::write_all(output, data),
            IdxData::I16 { data } => IdxDataWarp::<i16>::write_all(output, data),
            IdxData::I32 { data } => IdxDataWarp::<i32>::write_all(output, data),
            IdxData::F32 { data } => IdxDataWarp::<f32>::write_all(output, data),
            IdxData::F64 { data } => IdxDataWarp::<f64>::write_all(output, data),
        }
    }
}

impl Debug for IdxData {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        let t = match self {
            IdxData::U8 { .. } => "u8",
            IdxData::I8 { .. } => "i8",
            IdxData::I16 { .. } => "i16",
            IdxData::I32 { .. } => "i32",
            IdxData::F32 { .. } => "f32",
            IdxData::F64 { .. } => "f64",
        };
        f.write_str(t)?;
        f.write_str("[")?;
        self.data_len().fmt(f)?;
        f.write_str("]")?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct IdxFile {
    pub dimensions: Vec<u32>,
    pub data: IdxData,
}

impl IdxFile {
    fn dim_to_size(dim: &[u32]) -> usize {
        dim.iter()
            .copied()
            .map(|x| x as usize)
            .reduce(|a, b| a * b)
            .unwrap_or(0)
    }
    pub fn read<R: Read>(input: &mut R) -> Result<Self, io::Error> {
        let magic = input.read_u16::<byteorder::BE>()?;
        if magic != 0 {
            return Err(io::Error::new(ErrorKind::InvalidData, "magic"));
        }
        let idx_type = input.read_u8()?;

        let dim_len = input.read_u8()? as usize;
        let mut dim = Vec::with_capacity(dim_len);
        for _ in 0..dim_len {
            dim.push(input.read_u32::<byteorder::BE>()?);
        }
        let size = Self::dim_to_size(dim.deref());
        let data = IdxData::read(idx_type, input, size)?;

        Ok(IdxFile {
            dimensions: dim,
            data,
        })
    }

    pub fn write<W: Write>(&self, output: &mut W) -> Result<(), io::Error> {
        let len = self.dimensions.len();
        if len > u8::MAX as usize {
            return Err(io::Error::new(ErrorKind::InvalidData, "dim len"));
        }
        let size = Self::dim_to_size(self.dimensions.deref());
        if size != self.data.data_len() {
            return Err(io::Error::new(ErrorKind::InvalidData, "data len"));
        }
        let dim_len = len as u8;

        output.write_u16::<byteorder::BE>(0)?;
        output.write_u8(self.data.get_type())?;
        output.write_u8(dim_len)?;
        for &d in &self.dimensions {
            output.write_u32::<byteorder::BE>(d)?;
        }
        self.data.write(output)
    }

    pub fn read_file<P: AsRef<Path>>(path: P) -> Result<Self, io::Error> {
        let mut f = BufReader::new(File::open(path)?);
        Self::read(&mut f)
    }

    pub fn write_file<P: AsRef<Path>>(&self, path: P) -> Result<(), io::Error> {
        let mut f = BufWriter::new(File::create(path)?);
        self.write(&mut f)
    }
}
