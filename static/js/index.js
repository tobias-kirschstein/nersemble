// window.HELP_IMPROVE_VIDEOJS = false;
//
// var INTERP_BASE = "./static/interpolation/stacked";
// var NUM_INTERP_FRAMES = 240;
//
// var interp_images = [];
// function preloadInterpolationImages() {
//   for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
//     var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
//     interp_images[i] = new Image();
//     interp_images[i].src = path;
//   }
// }
//
// function setInterpolationImage(i) {
//   var image = interp_images[i];
//   image.ondragstart = function() { return false; };
//   image.oncontextmenu = function() { return false; };
//   $('#interpolation-image-wrapper').empty().append(image);
// }

var hashTableImages = {};
function preloadHashTableImages() {
    const targets = [-0.75, -0.5, -0.25, "0.0", 0.25, 0.5, 0.75];

    let key;
    let imageSrc;
    for (let beta1 of targets) {
        for (let beta2 of targets) {
            for (let beta3 of targets) {
                key = beta1 + "_" + beta2 + "_" + beta3
                imageSrc = "./static/images/blend_weight_interpolations/" + key + ".jpeg"
                hashTableImages[key] = new Image();
                hashTableImages[key].src = imageSrc
            }
        }
    }
}

function computeBlendWeightColor(beta) {
    positiveColor = $.Color("#E06666");
    negativeColor = $.Color("#0095dd");

    if (beta < 0) {
        blendColor = negativeColor;
    } else {
        blendColor = positiveColor;
    }
    white = $.Color("#FFF");
    alpha = Math.abs(beta) / 0.75;

    blendedColor = blendColor.alpha(alpha).blend(white.alpha(1 - alpha));

    return blendedColor;
}

function updateHashTableImage() {
    var beta1 = $("#slider-hash-table-1").val();
    var beta2 = $("#slider-hash-table-2").val();
    var beta3 = $("#slider-hash-table-3").val();

    beta1 = beta1 == 0 ? "0.0" : beta1;
    beta2 = beta2 == 0 ? "0.0" : beta2;
    beta3 = beta3 == 0 ? "0.0" : beta3;

    key = beta1 + "_" + beta2 + "_" + beta3;

    var image = hashTableImages[key];
    image.ondragstart = function() { return false; };
    image.oncontextmenu = function() { return false; };
    $('#hash-table-image-wrapper').empty().append(image);

    // Update vector
    $("#blend-weights-vector-1").css({"backgroundColor": computeBlendWeightColor(beta1)});
    $("#blend-weights-vector-2").css({"backgroundColor": computeBlendWeightColor(beta2)});
    $("#blend-weights-vector-3").css({"backgroundColor": computeBlendWeightColor(beta3)});
}


$(document).ready(function() {
    preloadHashTableImages();
    updateHashTableImage();

    var carousels = bulmaCarousel.attach('#results-carousel', {
        slidesToScroll: 1,
        slidesToShow: 1,
        loop: true,
        infinite: true,
        autoplay: true,
        autoplaySpeed: 8000,
    });

    // Start playing next video in carousel and pause previous video to limit load on browser
    for(var i = 0; i < carousels.length; i++) {
        // Add listener to  event
        carousels[i].on('before:show', state => {
            var nextId = (state.next + state.length) % state.length;  // state.next can be -1 or larger than the number of videos
            var nextVideoElement = $("#results-carousel .slider-item[data-slider-index='" + nextId + "'] video")[0];
            var previousVideoElement = $("#results-carousel .slider-item[data-slider-index='" + state.index + "'] video")[0];

            previousVideoElement.pause();
            previousVideoElement.currentTime = 0;
            nextVideoElement.currentTime = 0;
            nextVideoElement.play();
        });
    }


    $("#slider-hash-table-1, #slider-hash-table-2, #slider-hash-table-3").on("input", function () {
        updateHashTableImage();
    });

    // // Check for click events on the navbar burger icon
    // $(".navbar-burger").click(function() {
    //   // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
    //   $(".navbar-burger").toggleClass("is-active");
    //   $(".navbar-menu").toggleClass("is-active");
    //
    // });
    //
    // var options = {
	// 		slidesToScroll: 1,
	// 		slidesToShow: 3,
	// 		loop: true,
	// 		infinite: true,
	// 		autoplay: false,
	// 		autoplaySpeed: 3000,
    // }
    //
	// 	// Initialize all div with carousel class
    // var carousels = bulmaCarousel.attach('.carousel', options);
    //
    // // Loop on each carousel initialized
    // for(var i = 0; i < carousels.length; i++) {
    // 	// Add listener to  event
    // 	carousels[i].on('before:show', state => {
    // 		console.log(state);
    // 	});
    // }
    //
    // // Access to bulmaCarousel instance of an element
    // var element = document.querySelector('#my-element');
    // if (element && element.bulmaCarousel) {
    // 	// bulmaCarousel instance is available as element.bulmaCarousel
    // 	element.bulmaCarousel.on('before-show', function(state) {
    // 		console.log(state);
    // 	});
    // }
    //
    // /*var player = document.getElementById('interpolation-video');
    // player.addEventListener('loadedmetadata', function() {
    //   $('#interpolation-slider').on('input', function(event) {
    //     console.log(this.value, player.duration);
    //     player.currentTime = player.duration / 100 * this.value;
    //   })
    // }, false);*/
    // // preloadInterpolationImages();
    //
    // // $('#interpolation-slider').on('input', function(event) {
    // //   setInterpolationImage(this.value);
    // // });
    // // setInterpolationImage(0);
    // // $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);
    //
    // bulmaSlider.attach();

})