import sys
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import pyds


# ============================================================
# Configuração
# ============================================================

INPUT_VIDEO = "videos/fishing.mp4"


# ============================================================
# Callback executado para cada frame após a inferência/tracking
# ============================================================

def osd_sink_pad_buffer_probe(pad, info, user_data):

    gst_buffer = info.get_buffer()

    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    # Recupera metadados do DeepStream
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(
        hash(gst_buffer)
    )

    l_frame = batch_meta.frame_meta_list

    total_fish = 0

    while l_frame is not None:

        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num

        # Percorre objetos detectados
        l_obj = frame_meta.obj_meta_list

        while l_obj is not None:

            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            # ID da classe definido pelo seu modelo
            class_id = obj_meta.class_id

            # Exemplo:
            # 0 = fish
            if class_id == 0:

                total_fish += 1

                # Bounding box
                rect = obj_meta.rect_params

                x = rect.left
                y = rect.top
                width = rect.width
                height = rect.height

                confidence = obj_meta.confidence

                # Tracking ID
                track_id = obj_meta.object_id

                print(
                    f"[FRAME {frame_number}] "
                    f"Fish "
                    f"ID={track_id} "
                    f"confidence={confidence:.2f} "
                    f"bbox=({x:.1f}, {y:.1f}, "
                    f"{width:.1f}, {height:.1f})"
                )

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        print(
            f"[FRAME {frame_number}] "
            f"Fish detectados: {total_fish}"
        )

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


# ============================================================
# Criação de elemento GStreamer
# ============================================================

def make_element(factory_name, name):

    element = Gst.ElementFactory.make(factory_name, name)

    if not element:
        raise RuntimeError(
            f"Não foi possível criar: {factory_name}"
        )

    return element


# ============================================================
# Pipeline
# ============================================================

def create_pipeline():

    pipeline = Gst.Pipeline.new("fish-pipeline")

    # --------------------------------------------------------
    # Entrada
    # --------------------------------------------------------

    source = make_element(
        "filesrc",
        "source"
    )

    source.set_property(
        "location",
        INPUT_VIDEO
    )

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    decoder = make_element(
        "decodebin",
        "decoder"
    )

    # --------------------------------------------------------
    # Stream mux
    # --------------------------------------------------------

    streammux = make_element(
        "nvstreammux",
        "stream-muxer"
    )

    streammux.set_property(
        "batch-size",
        1
    )

    streammux.set_property(
        "width",
        1920
    )

    streammux.set_property(
        "height",
        1080
    )

    streammux.set_property(
        "batched-push-timeout",
        40000
    )

    # --------------------------------------------------------
    # Inferência
    # --------------------------------------------------------

    pgie = make_element(
        "nvinfer",
        "primary-inference"
    )

    pgie.set_property(
        "config-file-path",
        "config/config_infer.txt"
    )

    # --------------------------------------------------------
    # Tracker
    # --------------------------------------------------------

    tracker = make_element(
        "nvtracker",
        "tracker"
    )

    tracker.set_property(
        "tracker-width",
        640
    )

    tracker.set_property(
        "tracker-height",
        384
    )

    # --------------------------------------------------------
    # OSD
    # --------------------------------------------------------

    nvdsosd = make_element(
        "nvdsosd",
        "onscreendisplay"
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    sink = make_element(
        "nveglglessink",
        "video-output"
    )

    # --------------------------------------------------------
    # Adiciona elementos
    # --------------------------------------------------------

    elements = [
        source,
        decoder,
        streammux,
        pgie,
        tracker,
        nvdsosd,
        sink
    ]

    for element in elements:
        pipeline.add(element)

    # --------------------------------------------------------
    # filesrc → decodebin
    # --------------------------------------------------------

    source.link(decoder)

    # decodebin possui pad dinâmico
    def on_pad_added(decodebin, pad):

        caps = pad.get_current_caps()

        if not caps:
            return

        structure = caps.get_structure(0)

        name = structure.get_name()

        if name.startswith("video/"):

            sink_pad = streammux.get_request_pad(
                "sink_0"
            )

            if not sink_pad:
                raise RuntimeError(
                    "Não foi possível obter sink_0"
                )

            if sink_pad.is_linked():
                return

            pad.link(sink_pad)

    decoder.connect(
        "pad-added",
        on_pad_added
    )

    # --------------------------------------------------------
    # Pipeline principal
    # --------------------------------------------------------

    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(nvdsosd)
    nvdsosd.link(sink)

    # --------------------------------------------------------
    # Probe para acessar metadata
    # --------------------------------------------------------

    osd_sink_pad = nvdsosd.get_static_pad(
        "sink"
    )

    if not osd_sink_pad:

        raise RuntimeError(
            "Não foi possível obter OSD sink pad"
        )

    osd_sink_pad.add_probe(
        Gst.PadProbeType.BUFFER,
        osd_sink_pad_buffer_probe,
        None
    )

    return pipeline


# ============================================================
# Main
# ============================================================

def main():

    Gst.init(None)

    pipeline = create_pipeline()

    loop = GLib.MainLoop()

    bus = pipeline.get_bus()

    bus.add_signal_watch()

    def bus_call(bus, message, loop):

        msg_type = message.type

        if msg_type == Gst.MessageType.EOS:

            print("Fim do vídeo.")

            loop.quit()

        elif msg_type == Gst.MessageType.ERROR:

            error, debug = message.parse_error()

            print(
                "Erro:",
                error
            )

            print(
                "Debug:",
                debug
            )

            loop.quit()

        return True

    bus.connect(
        "message",
        bus_call,
        loop
    )

    print(
        "Iniciando pipeline DeepStream..."
    )

    pipeline.set_state(
        Gst.State.PLAYING
    )

    try:
        loop.run()

    except KeyboardInterrupt:
        pass

    finally:

        pipeline.set_state(
            Gst.State.NULL
        )


if __name__ == "__main__":
    main()
