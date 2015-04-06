var Dropzone = React.createClass({
  getInitialState: function() {
    return {
      isDragActive: false
    }
  },

  propTypes: {
    onDrop: React.PropTypes.func.isRequired,
    size: React.PropTypes.number,
    style: React.PropTypes.object
  },

  onDragLeave: function(e) {
    this.setState({
      isDragActive: false
    });
  },

  onDragOver: function(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";

    this.setState({
      isDragActive: true
    });
  },

  onDrop: function(e) {
    e.preventDefault();

    this.setState({
      isDragActive: false
    });

    var files;
    if (e.dataTransfer) {
      files = e.dataTransfer.files;
    } else if (e.target) {
      files = e.target.files;
    }

    if (this.props.onDrop) {
      files = Array.prototype.slice.call(files);
      this.props.onDrop(files);
    }
  },

  onClick: function () {
    this.refs.fileInput.getDOMNode().click();
  },

  render: function() {

    var className = this.props.className || 'dropzone';
    if (this.state.isDragActive) {
      className += ' active';
    };

    var style = this.props.style || {
      width: this.props.size || 100,
      height: this.props.size || 100,
      borderStyle: this.state.isDragActive ? "solid" : "dashed"
    };

    if (this.props.className) {
      style = this.props.style;
    }

    return (
      <div className={className} style={style} onClick={this.onClick} onDragLeave={this.onDragLeave} onDragOver={this.onDragOver} onDrop={this.onDrop}>
        <input style={{display: 'none' }} type='file' multiple ref='fileInput' onChange={this.onDrop} />
        {this.props.children}
      </div>
    );
  }

});

var DropzoneImage = React.createClass({displayName: "DropzoneDemo",
    getInitialState: function () {
      return {
        files: []
      };
    },

    onDrop: function (files) {
      console.log('Received files: ', files);

      var file = files[0];
      var reader = new FileReader();
      reader.onload = function(upload){
        this.setState({
              files: files,
              data_uri: reader.result
        });
        var output = document.getElementById('output');
        output.src = this.state.data_uri;
        console.log(this.state.data_uri)

        this.props.onDrop({data_uri: this.state.data_uri});
      }.bind(this);

      reader.readAsDataURL(file);
    },

    showFiles: function () {
      if (this.state.files.length <= 0) {
        return '';
      };

      var files = this.state.files;

      return (
        React.createElement("div", null, 
          React.createElement("h3", null, "Dropped files: "), 
          React.createElement("ul", null, 
            [].map.call(files, function (f, i) {
              return React.createElement("li", {key: i}, f.name + ' : ' + f.size + ' bytes.')
            })
          )
        )
        );
    },

    render: function () {
      var styling = {
        padding: 30
      };

      return (
          React.createElement("div", null, 
            React.createElement(Dropzone, {onDrop: this.onDrop, size: 150}, 
              React.createElement("div", {style: styling}, "Try dropping some files here, or click to select files to upload.")
            ), 
            this.showFiles()
          )
      );
    }
});

var ImageGrid = React.createClass({
  render: function() {
    var imageNodes = this.props.images.map(function(image, index) {
      return (
         React.createElement("img", {src: image, style:{margin:"5px 5px"}})
      );
    });

    console.log({imageNodes})
    return (
      <div className="imageGrid">
        {imageNodes}
      </div>
    );
  }
});

var CommentForm = React.createClass({
  componentDidMount: function() {
    // do something after the component mounts but before rendering
  },
  getInitialState: function() {
    return {images: ['../examples/cat.jpg','../examples/cat.jpg','../examples/cat.jpg']};
  },
  handleSubmit: function(e) {
    e.preventDefault();
    console.log('submit  button clicked!');

    $.ajax({
        url: this.props.url,
        dataType: 'json',
        type: 'POST',
        data: {image_uri: this.state.image_uri},
        success: function(data) {
          this.setState({data: data});
        }.bind(this),
        error: function(xhr, status, err) {
          console.error(this.props.url, status, err.toString());
        }.bind(this)
      });

  },

  handleOnDrop: function(drop)
  {
    this.setState({image_uri: drop.data_uri});
  },

  render: function() {
    return (
      <form className="commentForm" onSubmit={this.handleSubmit}>
        <DropzoneImage onDrop={this.handleOnDrop}/>
        <input type="submit" value="Submit" />
        <ImageGrid images={this.state.images}/>
      </form>
    );
  }
});

React.render(<CommentForm url="image.json"/>, document.getElementById('content'));

