import logging
import os
import xml.etree.ElementTree as ET

import tifffile as tiff

logger = logging.getLogger(__name__)


def parse_metadata(filename):
    file_name = os.path.basename(filename)
    name = os.path.splitext(file_name)[0]
    metadata = {}

    with tiff.TiffFile(filename) as tif:
        raw_meta_data = {}
        page = tif.pages[0]
        if isinstance(page, tiff.TiffFrame):
            page = page.aspage()
        for tag in page.tags.values():
            raw_meta_data[tag.name] = tag.value

    try:
        desc = raw_meta_data.get("ImageDescription")
        if desc:
            root = ET.fromstring(desc)
            namespace_uri = root.tag[root.tag.find(
                "{") + 1: root.tag.find("}")]
            ns = {"ome": namespace_uri}
            pixels = root.find(".//ome:Pixels", namespaces=ns)

            if pixels is not None:
                metadata = {
                    "Name": name,
                    "URI": filename,
                    "Width": pixels.attrib.get("SizeX"),
                    "Height": pixels.attrib.get("SizeY"),
                    "Dimension (CZT)": f"{pixels.attrib.get('SizeC')} x {pixels.attrib.get('SizeZ')} x {pixels.attrib.get('SizeT')}",
                    "Pixel Type": pixels.attrib.get("Type"),
                    "PhysicalSizeX": f"{pixels.attrib.get('PhysicalSizeX')} {pixels.attrib.get('PhysicalSizeXUnit')}",
                    "PhysicalSizeY": f"{pixels.attrib.get('PhysicalSizeY')} {pixels.attrib.get('PhysicalSizeYUnit')}",
                    "DimensionOrder": pixels.attrib.get("DimensionOrder"),
                }
            else:
                logger.warning("Pixels element not found.")
        else:
            logger.warning("ImageDescription tag not found.")

    except ET.ParseError as e:
        logger.error("Parse error has occurred: %s", e)
    except Exception as e:
        logger.error(f"Metadata parsing error: {e}")

    if not metadata:
        metadata["Name"] = name
        metadata["URI"] = filename
        metadata["Width"] = raw_meta_data.get("ImageWidth", "Unknown")
        metadata["Height"] = raw_meta_data.get("ImageLength", "Unknown")
        bps = raw_meta_data.get('BitsPerSample')
        metadata["Pixel Type"] = f"uint{bps}" if bps else "Unknown"
        metadata["Dimension (CZT)"] = "Unknown"
        metadata["PhysicalSizeX"] = "Unknown"
        metadata["PhysicalSizeY"] = "Unknown"
        metadata["DimensionOrder"] = "Unknown"

    return metadata

