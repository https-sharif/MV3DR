() => {
    const el = document.getElementById("model-container");
    if (!el) {
        console.error("Model container not found");
        return;
    }

    if (document.fullscreenElement) {
        document.exitFullscreen();
    } else {
        if (el.requestFullscreen) {
            el.requestFullscreen();
        } else if (el.webkitRequestFullscreen) {
            el.webkitRequestFullscreen();
        } else if (el.msRequestFullscreen) {
            el.msRequestFullscreen();
        }
    }
}
