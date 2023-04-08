use std::{fs::{File, create_dir_all}, io::{BufReader, Write}, path::PathBuf};
mod parse_mediawiki_dump;

fn main() {
    let f = File::open("../wiki.xml");
    match f {
        Ok(file) =>parse(BufReader::new(file)),
        Err(error)=>println!("Error reading file: {}", error)
    }
}

fn parse(source: impl std::io::BufRead) {
    // total page number = 22781695, very large
    let parser = parse_mediawiki_dump::parse(source);
    for (i,result) in parser.take(10).enumerate() {
        match result {
            Err(error) => {
                eprintln!("Error: {}", error);
                std::process::exit(1);
            }
            Ok(page) => {
                let text = &page.text;
                let path = format!("../content/{i}");
                let p = PathBuf::from(&path);
                if !p.is_dir(){
                    create_dir_all(&path).unwrap();
                    let json = serde_json::to_string(&page).unwrap();
                    let mut jsonfile = File::create(format!("{path}/{i}.json")).unwrap();
                    let mut txtfile = File::create(format!("{path}/{i}.txt")).unwrap();
                    jsonfile.write_all(json.as_bytes()).unwrap();
                    txtfile.write_all(text.as_bytes()).unwrap();
                }
            }
        }
    }
}