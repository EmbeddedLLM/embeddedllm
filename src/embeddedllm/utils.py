import uuid
import base64
import filetype
import re

def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def decode_base64(data_uri:str):
    # Extract the base64 part from the data URI
    base64_string = re.sub(r'^data:.*;base64,', '', data_uri)
    # Decode the base64 string
    file_data = base64.b64decode(base64_string)

    # Get the file type
    kind = filetype.guess(file_data)

    if kind is None:
        return file_data, None
    else:
        return file_data, kind.mime


if __name__ == "__main__":
        # Example base64 string for a PNG image
        base64_string = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wcA"
            "AwAB/6F8n9kAAAAASUVORK5CYII="
        )

        expected_mime_type = "image/png"

        # Call the function
        result = decode_base64(base64_string)

        print(result)