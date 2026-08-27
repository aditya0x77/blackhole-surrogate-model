#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <fstream>
#include <sstream>
#include <random>
#include <filesystem>
#include <limits>
#include <algorithm>
#include <cstdio>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
using namespace glm;
using namespace std;
using Clock = std::chrono::high_resolution_clock;

// VARS
double lastPrintTime = 0.0;
int    framesCount   = 0;
double c = 299792458.0;
double G = 6.67430e-11;
struct Ray;
bool Gravity = false;

struct Camera {
    // Center the camera orbit on the black hole at (0, 0, 0)
    vec3 target = vec3(0.0f, 0.0f, 0.0f); // Always look at the black hole center
    float radius = 6.34194e10f;
    float minRadius = 1e10f, maxRadius = 1e12f;

    float azimuth = 0.0f;
    float elevation = M_PI / 2.0f;

    float orbitSpeed = 0.01f;
    float panSpeed = 0.01f;
    double zoomSpeed = 25e9f;

    bool dragging = false;
    bool panning = false;
    bool moving = false; // For compute shader optimization
    double lastX = 0.0, lastY = 0.0;

    // Calculate camera position in world space
    vec3 position() const {
        float clampedElevation = glm::clamp(elevation, 0.01f, float(M_PI) - 0.01f);
        // Orbit around (0,0,0) always
        return vec3(
            radius * sin(clampedElevation) * cos(azimuth),
            radius * cos(clampedElevation),
            radius * sin(clampedElevation) * sin(azimuth)
        );
    }
    void update() {
        // Always keep target at black hole center
        target = vec3(0.0f, 0.0f, 0.0f);
        if(dragging | panning) {
            moving = true;
        } else {
            moving = false;
        }
    }

    void processMouseMove(double x, double y) {
        float dx = float(x - lastX);
        float dy = float(y - lastY);

        if (dragging && panning) {
            // Pan: Shift + Left or Middle Mouse
            // Disable panning to keep camera centered on black hole
        }
        else if (dragging && !panning) {
            // Orbit: Left mouse only
            azimuth   += dx * orbitSpeed;
            elevation -= dy * orbitSpeed;
            elevation = glm::clamp(elevation, 0.01f, float(M_PI) - 0.01f);
        }

        lastX = x;
        lastY = y;
        update();
    }
    void processMouseButton(int button, int action, int mods, GLFWwindow* win) {
        if (button == GLFW_MOUSE_BUTTON_LEFT || button == GLFW_MOUSE_BUTTON_MIDDLE) {
            if (action == GLFW_PRESS) {
                dragging = true;
                // Disable panning so camera always orbits center
                panning = false;
                glfwGetCursorPos(win, &lastX, &lastY);
            } else if (action == GLFW_RELEASE) {
                dragging = false;
                panning = false;
            }
        }
        if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            if (action == GLFW_PRESS) {
                Gravity = true;
            } else if (action == GLFW_RELEASE) {
                Gravity = false;
            }
        }
    }
    void processScroll(double xoffset, double yoffset) {
        radius -= yoffset * zoomSpeed;
        radius = glm::clamp(radius, minRadius, maxRadius);
        update();
    }
    void processKey(int key, int scancode, int action, int mods) {
        if (action == GLFW_PRESS && key == GLFW_KEY_G) {
            Gravity = !Gravity;
            cout << "[INFO] Gravity turned " << (Gravity ? "ON" : "OFF") << endl;
        }
    }
};
Camera camera;

struct BlackHole {
    vec3 position;
    double mass;
    double radius;
    double r_s;

    BlackHole(vec3 pos, float m) : position(pos), mass(m) {r_s = 2.0 * G * mass / (c*c);}
    bool Intercept(float px, float py, float pz) const {
        double dx = double(px) - double(position.x);
        double dy = double(py) - double(position.y);
        double dz = double(pz) - double(position.z);
        double dist2 = dx * dx + dy * dy + dz * dz;
        return dist2 < r_s * r_s;
    }
};
BlackHole SagA(vec3(0.0f, 0.0f, 0.0f), 8.54e36); // Sagittarius A black hole
struct ObjectData {
    vec4 posRadius; // xyz = position, w = radius
    vec4 color;     // rgb = color, a = unused
    float  mass;
    vec3 velocity = vec3(0.0f, 0.0f, 0.0f); // Initial velocity
};
vector<ObjectData> objects = {
    { vec4(4e11f, 0.0f, 0.0f, 4e10f)   , vec4(1,1,0,1), 1.98892e30 },
    { vec4(0.0f, 0.0f, 4e11f, 4e10f)   , vec4(1,0,0,1), 1.98892e30 },
    { vec4(0.0f, 0.0f, 0.0f, SagA.r_s) , vec4(0,0,0,1), static_cast<float>(SagA.mass)  },
    //{ vec4(6e10f, 0.0f, 0.0f, 5e10f), vec4(0,1,0,1) }
};

// Sets the black hole's mass and recomputes its Schwarzschild radius.
// Used by the dataset generator to vary mass (and therefore ring size) per sample.
void setBlackHoleMass(double mass) {
    SagA.mass = mass;
    SagA.r_s  = 2.0 * G * SagA.mass / (c * c);
}

struct Engine {
    GLuint gridShaderProgram;
    // -- Quad & Texture render -- //
    GLFWwindow* window;
    GLuint quadVAO;
    GLuint texture;
    GLuint shaderProgram;
    GLuint computeProgram = 0;
    // -- UBOs -- //
    GLuint cameraUBO = 0;
    GLuint diskUBO = 0;
    GLuint objectsUBO = 0;
    // -- grid mess vars -- //
    GLuint gridVAO = 0;
    GLuint gridVBO = 0;
    GLuint gridEBO = 0;
    int gridIndexCount = 0;

    int WIDTH = 800;  // Window width
    int HEIGHT = 600; // Window height
    int COMPUTE_WIDTH  = 200;   // Compute resolution width
    int COMPUTE_HEIGHT = 150;  // Compute resolution height
    float width = 100000000000.0f; // Width of the viewport in meters
    float height = 75000000000.0f; // Height of the viewport in meters
    
    Engine() {
        if (!glfwInit()) {
            cerr << "GLFW init failed\n";
            exit(EXIT_FAILURE);
        }
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Black Hole", nullptr, nullptr);
        if (!window) {
            cerr << "Failed to create GLFW window\n";
            glfwTerminate();
            exit(EXIT_FAILURE);
        }
        glfwMakeContextCurrent(window);
        glewExperimental = GL_TRUE;
        GLenum glewErr = glewInit();
        if (glewErr != GLEW_OK) {
            cerr << "Failed to initialize GLEW: "
                << (const char*)glewGetErrorString(glewErr)
                << "\n";
            glfwTerminate();
            exit(EXIT_FAILURE);
        }
        cout << "OpenGL " << glGetString(GL_VERSION) << "\n";
        this->shaderProgram = CreateShaderProgram();
        gridShaderProgram = CreateShaderProgram("grid.vert", "grid.frag");

        computeProgram = CreateComputeProgram("geodesic.comp");
        glGenBuffers(1, &cameraUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, cameraUBO);
        glBufferData(GL_UNIFORM_BUFFER, 128, nullptr, GL_DYNAMIC_DRAW); // alloc ~128 bytes
        glBindBufferBase(GL_UNIFORM_BUFFER, 1, cameraUBO); // binding = 1 matches shader

        glGenBuffers(1, &diskUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, diskUBO);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(float) * 8, nullptr, GL_DYNAMIC_DRAW); // r1, r2, num, thickness, bh_rs + padding
        glBindBufferBase(GL_UNIFORM_BUFFER, 2, diskUBO); // binding = 2 matches compute shader

        glGenBuffers(1, &objectsUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, objectsUBO);
        // allocate space for 16 objects: 
        // sizeof(int) + padding + 16×(vec4 posRadius + vec4 color)
        GLsizeiptr objUBOSize = sizeof(int) + 3 * sizeof(float)
            + 16 * (sizeof(vec4) + sizeof(vec4))
            + 16 * sizeof(float); // 16 floats for mass
        glBufferData(GL_UNIFORM_BUFFER, objUBOSize, nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, 3, objectsUBO);  // binding = 3 matches shader

        auto result = QuadVAO();
        this->quadVAO = result[0];
        this->texture = result[1];
    }
    void generateGrid(const vector<ObjectData>& objects) {
        const int gridSize = 25;
        const float spacing = 1e10f;  // tweak this

        vector<vec3> vertices;
        vector<GLuint> indices;

        for (int z = 0; z <= gridSize; ++z) {
            for (int x = 0; x <= gridSize; ++x) {
                float worldX = (x - gridSize / 2) * spacing;
                float worldZ = (z - gridSize / 2) * spacing;

                float y = 0.0f;

                // ✅ Warp grid using Schwarzschild geometry
                for (const auto& obj : objects) {
                    vec3 objPos = vec3(obj.posRadius);
                    double mass = obj.mass;
                    double radius = obj.posRadius.w;

                    double r_s = 2.0 * G * mass / (c * c);
                    double dx = worldX - objPos.x;
                    double dz = worldZ - objPos.z;
                    double dist = sqrt(dx * dx + dz * dz);

                    // prevent sqrt of negative or divide-by-zero (inside or at the black hole center)
                    if (dist > r_s) {
                        double deltaY = 2.0 * sqrt(r_s * (dist - r_s));
                        y += static_cast<float>(deltaY) - 3e10f;
                    } else {
                        // 🔴 For points inside or at r_s: make it dip down sharply
                        y += 2.0f * static_cast<float>(sqrt(r_s * r_s)) - 3e10f;  // or add a deep pit
                    }
                }

                vertices.emplace_back(worldX, y, worldZ);
            }
        }

        // 🧩 Add indices for GL_LINE rendering
        for (int z = 0; z < gridSize; ++z) {
            for (int x = 0; x < gridSize; ++x) {
                int i = z * (gridSize + 1) + x;
                indices.push_back(i);
                indices.push_back(i + 1);

                indices.push_back(i);
                indices.push_back(i + gridSize + 1);
            }
        }

        // 🔌 Upload to GPU
        if (gridVAO == 0) glGenVertexArrays(1, &gridVAO);
        if (gridVBO == 0) glGenBuffers(1, &gridVBO);
        if (gridEBO == 0) glGenBuffers(1, &gridEBO);

        glBindVertexArray(gridVAO);

        glBindBuffer(GL_ARRAY_BUFFER, gridVBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vec3), vertices.data(), GL_DYNAMIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gridEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(0); // location = 0
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vec3), (void*)0);

        gridIndexCount = indices.size();

        glBindVertexArray(0);
    }
    void drawGrid(const mat4& viewProj) {
        glUseProgram(gridShaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(gridShaderProgram, "viewProj"),
                        1, GL_FALSE, glm::value_ptr(viewProj));
        glBindVertexArray(gridVAO);

        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glDrawElements(GL_LINES, gridIndexCount, GL_UNSIGNED_INT, 0);

        glBindVertexArray(0);
        glEnable(GL_DEPTH_TEST);
    }
    void drawFullScreenQuad() {
        glUseProgram(shaderProgram); // fragment + vertex shader
        glBindVertexArray(quadVAO);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        glUniform1i(glGetUniformLocation(shaderProgram, "screenTexture"), 0);

        glDisable(GL_DEPTH_TEST);  // draw as background
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 6);  // 2 triangles
        glEnable(GL_DEPTH_TEST);
    }
    GLuint CreateShaderProgram(){
        const char* vertexShaderSource = R"(
        #version 330 core
        layout (location = 0) in vec2 aPos;  // Changed to vec2
        layout (location = 1) in vec2 aTexCoord;
        out vec2 TexCoord;
        void main() {
            gl_Position = vec4(aPos, 0.0, 1.0);  // Explicit z=0
            TexCoord = aTexCoord;
        })";

        const char* fragmentShaderSource = R"(
        #version 330 core
        in vec2 TexCoord;
        out vec4 FragColor;
        uniform sampler2D screenTexture;
        void main() {
            FragColor = texture(screenTexture, TexCoord);
        })";

        // vertex shader
        GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
        glCompileShader(vertexShader);

        // fragment shader
        GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
        glCompileShader(fragmentShader);

        GLuint shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);

        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        return shaderProgram;
    };
    GLuint CreateShaderProgram(const char* vertPath, const char* fragPath) {
        auto loadShader = [](const char* path, GLenum type) -> GLuint {
            std::ifstream in(path);
            if (!in.is_open()) {
                std::cerr << "Failed to open shader: " << path << "\n";
                exit(EXIT_FAILURE);
            }
            std::stringstream ss;
            ss << in.rdbuf();
            std::string srcStr = ss.str();
            const char* src = srcStr.c_str();

            GLuint shader = glCreateShader(type);
            glShaderSource(shader, 1, &src, nullptr);
            glCompileShader(shader);

            GLint success;
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if (!success) {
                GLint logLen;
                glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLen);
                std::vector<char> log(logLen);
                glGetShaderInfoLog(shader, logLen, nullptr, log.data());
                std::cerr << "Shader compile error (" << path << "):\n" << log.data() << "\n";
                exit(EXIT_FAILURE);
            }
            return shader;
        };

        GLuint vertShader = loadShader(vertPath, GL_VERTEX_SHADER);
        GLuint fragShader = loadShader(fragPath, GL_FRAGMENT_SHADER);

        GLuint program = glCreateProgram();
        glAttachShader(program, vertShader);
        glAttachShader(program, fragShader);
        glLinkProgram(program);

        GLint linkSuccess;
        glGetProgramiv(program, GL_LINK_STATUS, &linkSuccess);
        if (!linkSuccess) {
            GLint logLen;
            glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLen);
            std::vector<char> log(logLen);
            glGetProgramInfoLog(program, logLen, nullptr, log.data());
            std::cerr << "Shader link error:\n" << log.data() << "\n";
            exit(EXIT_FAILURE);
        }

        glDeleteShader(vertShader);
        glDeleteShader(fragShader);

        return program;
    }
    GLuint CreateComputeProgram(const char* path) {
        // 1) read GLSL source
        std::ifstream in(path);
        if(!in.is_open()) {
            std::cerr << "Failed to open compute shader: " << path << "\n";
            exit(EXIT_FAILURE);
        }
        std::stringstream ss;
        ss << in.rdbuf();
        std::string srcStr = ss.str();
        const char* src = srcStr.c_str();

        // 2) compile
        GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
        glShaderSource(cs, 1, &src, nullptr);
        glCompileShader(cs);
        GLint ok; 
        glGetShaderiv(cs, GL_COMPILE_STATUS, &ok);
        if(!ok) {
            GLint logLen;
            glGetShaderiv(cs, GL_INFO_LOG_LENGTH, &logLen);
            std::vector<char> log(logLen);
            glGetShaderInfoLog(cs, logLen, nullptr, log.data());
            std::cerr << "Compute shader compile error:\n" << log.data() << "\n";
            exit(EXIT_FAILURE);
        }

        // 3) link
        GLuint prog = glCreateProgram();
        glAttachShader(prog, cs);
        glLinkProgram(prog);
        glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        if(!ok) {
            GLint logLen;
            glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &logLen);
            std::vector<char> log(logLen);
            glGetProgramInfoLog(prog, logLen, nullptr, log.data());
            std::cerr << "Compute shader link error:\n" << log.data() << "\n";
            exit(EXIT_FAILURE);
        }

        glDeleteShader(cs);
        return prog;
    }
    void dispatchCompute(const Camera& cam) {
        // determine target compute‐res
        int cw = cam.moving ? COMPUTE_WIDTH  : 200;
        int ch = cam.moving ? COMPUTE_HEIGHT : 150;

        // 1) reallocate the texture if needed
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D,
                    0,                // mip
                    GL_RGBA8,         // internal format
                    cw,               // width
                    ch,               // height
                    0, GL_RGBA, 
                    GL_UNSIGNED_BYTE, 
                    nullptr);

        // 2) bind compute program & UBOs
        glUseProgram(computeProgram);
        uploadCameraUBO(cam, cw, ch);
        uploadDiskUBO(static_cast<float>(SagA.r_s) * 2.2f,
                      static_cast<float>(SagA.r_s) * 5.2f,
                      static_cast<float>(SagA.r_s));
        uploadObjectsUBO(objects);

        // 3) bind it as image unit 0
        glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);

        // 4) dispatch grid
        GLuint groupsX = (GLuint)std::ceil(cw / 16.0f);
        GLuint groupsY = (GLuint)std::ceil(ch / 16.0f);
        glDispatchCompute(groupsX, groupsY, 1);

        // 5) sync
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }
    // Offscreen render used by the dataset generator: lets caller fully control
    // resolution, disk extents, black hole radius and the object list per sample.
    void renderDatasetSample(const Camera& cam, int resW, int resH,
                              float diskR1, float diskR2, float bhRs,
                              const vector<ObjectData>& objs) {
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, resW, resH, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        glUseProgram(computeProgram);
        uploadCameraUBO(cam, resW, resH);
        uploadDiskUBO(diskR1, diskR2, bhRs);
        uploadObjectsUBO(objs);

        glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);

        GLuint groupsX = (GLuint)std::ceil(resW / 16.0f);
        GLuint groupsY = (GLuint)std::ceil(resH / 16.0f);
        glDispatchCompute(groupsX, groupsY, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        glFinish(); // make sure the compute pass is done before we read the texture back
    }
    // resW/resH is the resolution the compute shader should actually render at
    // (defaults to the window's aspect-driving WIDTH/HEIGHT for interactive mode,
    // but the dataset generator passes its own render resolution).
    void uploadCameraUBO(const Camera& cam, int resW, int resH) {
        struct UBOData {
            vec3 pos; float _pad0;
            vec3 right; float _pad1;
            vec3 up; float _pad2;
            vec3 forward; float _pad3;
            float tanHalfFov;
            float aspect;
            bool moving;
            int resW;
            int resH;
        } data;
        vec3 fwd = normalize(cam.target - cam.position());
        vec3 up = vec3(0, 1, 0); // y axis is up, so disk is in x-z plane
        vec3 right = normalize(cross(fwd, up));
        up = cross(right, fwd);

        data.pos = cam.position();
        data.right = right;
        data.up = up;
        data.forward = fwd;
        data.tanHalfFov = tan(radians(60.0f * 0.5f));
        data.aspect = float(resW) / float(resH);
        data.moving = cam.dragging || cam.panning;
        data.resW = resW;
        data.resH = resH;

        glBindBuffer(GL_UNIFORM_BUFFER, cameraUBO);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(UBOData), &data);
    }
    void uploadObjectsUBO(const vector<ObjectData>& objs) {
        struct UBOData {
            int   numObjects;
            float _pad0, _pad1, _pad2;        // <-- pad out to 16 bytes
            vec4  posRadius[16];
            vec4  color[16];
            float  mass[16]; 
        } data;

        size_t count = std::min(objs.size(), size_t(16));
        data.numObjects = static_cast<int>(count);

        for (size_t i = 0; i < count; ++i) {
            data.posRadius[i] = objs[i].posRadius;
            data.color[i] = objs[i].color;
            data.mass[i] = objs[i].mass;
        }

        // Upload
        glBindBuffer(GL_UNIFORM_BUFFER, objectsUBO);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(data), &data);
    }
    // r1/r2 = accretion disk inner/outer radius (meters), bhRs = current black hole
    // Schwarzschild radius. Explicit params let the dataset generator vary these
    // per-sample; interactive mode just passes SagA's current fixed ratios.
    void uploadDiskUBO(float r1, float r2, float bhRs) {
        float num = 2.0f;                // number of rays
        float thickness = 1e9f;          // padding for std140 alignment
        float diskData[8] = { r1, r2, num, thickness, bhRs, 0.0f, 0.0f, 0.0f };

        glBindBuffer(GL_UNIFORM_BUFFER, diskUBO);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(diskData), diskData);
    }
    
    vector<GLuint> QuadVAO(){
        float quadVertices[] = {
            // positions   // texCoords
            -1.0f,  1.0f,  0.0f, 1.0f,  // top left
            -1.0f, -1.0f,  0.0f, 0.0f,  // bottom left
            1.0f, -1.0f,  1.0f, 0.0f,  // bottom right

            -1.0f,  1.0f,  0.0f, 1.0f,  // top left
            1.0f, -1.0f,  1.0f, 0.0f,  // bottom right
            1.0f,  1.0f,  1.0f, 1.0f   // top right
        };
        
        GLuint VAO, VBO;
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);

        GLuint texture;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D,
                    0,             // mip
                    GL_RGBA8,      // internal format
                    COMPUTE_WIDTH,
                    COMPUTE_HEIGHT,
                    0,
                    GL_RGBA,
                    GL_UNSIGNED_BYTE,
                    nullptr);
        vector<GLuint> VAOtexture = {VAO, texture};
        return VAOtexture;
    }
    void renderScene() {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(shaderProgram);
        glBindVertexArray(quadVAO);
        // make sure your fragment shader samples from texture unit 0:
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glfwSwapBuffers(window);
        glfwPollEvents();
    };
};
Engine engine;
void setupCameraCallbacks(GLFWwindow* window) {
    glfwSetWindowUserPointer(window, &camera);

    glfwSetMouseButtonCallback(window, [](GLFWwindow* win, int button, int action, int mods) {
        Camera* cam = (Camera*)glfwGetWindowUserPointer(win);
        cam->processMouseButton(button, action, mods, win);
    });

    glfwSetCursorPosCallback(window, [](GLFWwindow* win, double x, double y) {
        Camera* cam = (Camera*)glfwGetWindowUserPointer(win);
        cam->processMouseMove(x, y);
    });

    glfwSetScrollCallback(window, [](GLFWwindow* win, double xoffset, double yoffset) {
        Camera* cam = (Camera*)glfwGetWindowUserPointer(win);
        cam->processScroll(xoffset, yoffset);
    });

    glfwSetKeyCallback(window, [](GLFWwindow* win, int key, int scancode, int action, int mods) {
        Camera* cam = (Camera*)glfwGetWindowUserPointer(win);
        cam->processKey(key, scancode, action, mods);
    });
}


// ==================== DATASET GENERATOR ==================== //
// Writes a real, standards-compliant PNG using only the standard library — no
// libpng/zlib dependency. It uses DEFLATE's "stored" (uncompressed) block type,
// which is a fully valid part of the zlib/PNG spec, just skipping the compression
// step. Files are bigger than a compressed PNG would be, but open correctly in
// any viewer, PIL, OpenCV, etc.
namespace PNGWriter {
    unsigned long crcTable[256];
    bool crcTableComputed = false;

    void makeCrcTable() {
        for (unsigned long n = 0; n < 256; ++n) {
            unsigned long cRc = n;
            for (int k = 0; k < 8; ++k) {
                cRc = (cRc & 1) ? (0xEDB88320UL ^ (cRc >> 1)) : (cRc >> 1);
            }
            crcTable[n] = cRc;
        }
        crcTableComputed = true;
    }
    unsigned long crc32(const unsigned char* buf, size_t len) {
        if (!crcTableComputed) makeCrcTable();
        unsigned long c = 0xffffffffUL;
        for (size_t n = 0; n < len; ++n) {
            c = crcTable[(c ^ buf[n]) & 0xff] ^ (c >> 8);
        }
        return c ^ 0xffffffffUL;
    }

    void pushBE32(std::vector<unsigned char>& out, unsigned long v) {
        out.push_back((unsigned char)((v >> 24) & 0xFF));
        out.push_back((unsigned char)((v >> 16) & 0xFF));
        out.push_back((unsigned char)((v >> 8) & 0xFF));
        out.push_back((unsigned char)(v & 0xFF));
    }

    void writeChunk(std::ofstream& out, const char* type, const std::vector<unsigned char>& data) {
        std::vector<unsigned char> lenBytes;
        pushBE32(lenBytes, (unsigned long)data.size());
        out.write(reinterpret_cast<char*>(lenBytes.data()), 4);

        std::vector<unsigned char> typeAndData(type, type + 4);
        typeAndData.insert(typeAndData.end(), data.begin(), data.end());
        out.write(reinterpret_cast<char*>(typeAndData.data()), (std::streamsize)typeAndData.size());

        std::vector<unsigned char> crcBytes;
        pushBE32(crcBytes, crc32(typeAndData.data(), typeAndData.size()));
        out.write(reinterpret_cast<char*>(crcBytes.data()), 4);
    }

    // Wraps raw bytes in a valid but uncompressed zlib stream (RFC1950 header +
    // RFC1951 "stored" deflate blocks + Adler32 trailer).
    std::vector<unsigned char> zlibStore(const std::vector<unsigned char>& raw) {
        std::vector<unsigned char> out;
        out.push_back(0x78); // CMF: deflate, 32K window
        out.push_back(0x01); // FLG: chosen so (0x78*256 + 0x01) % 31 == 0

        const size_t maxBlock = 65535;
        if (raw.empty()) {
            out.push_back(0x01); // BFINAL=1, BTYPE=00 (stored)
            out.push_back(0x00); out.push_back(0x00);
            out.push_back(0xFF); out.push_back(0xFF);
        } else {
            size_t pos = 0;
            while (pos < raw.size()) {
                size_t remaining = raw.size() - pos;
                size_t blockLen = std::min(remaining, maxBlock);
                bool isFinal = (pos + blockLen >= raw.size());

                out.push_back(isFinal ? 0x01 : 0x00);
                unsigned short len  = (unsigned short)blockLen;
                unsigned short nlen = (unsigned short)(~len);
                out.push_back((unsigned char)(len & 0xFF));
                out.push_back((unsigned char)((len >> 8) & 0xFF));
                out.push_back((unsigned char)(nlen & 0xFF));
                out.push_back((unsigned char)((nlen >> 8) & 0xFF));
                out.insert(out.end(), raw.begin() + pos, raw.begin() + pos + blockLen);
                pos += blockLen;
            }
        }

        unsigned long a = 1, b = 0;
        for (unsigned char byte : raw) {
            a = (a + byte) % 65521;
            b = (b + a) % 65521;
        }
        pushBE32(out, (b << 16) | a);
        return out;
    }

    void write(const std::string& path, int width, int height, const std::vector<unsigned char>& rgba) {
        // Raw PNG scanlines: 1 filter byte (0 = None) + RGB triples per row
        std::vector<unsigned char> raw;
        raw.reserve((size_t)(width * 3 + 1) * height);
        for (int y = 0; y < height; ++y) {
            raw.push_back(0);
            for (int x = 0; x < width; ++x) {
                size_t srcIdx = (size_t)(y * width + x) * 4; // source is RGBA
                raw.push_back(rgba[srcIdx + 0]); // R
                raw.push_back(rgba[srcIdx + 1]); // G
                raw.push_back(rgba[srcIdx + 2]); // B
            }
        }

        std::vector<unsigned char> idatData = zlibStore(raw);

        std::ofstream out(path, std::ios::binary);
        unsigned char sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
        out.write(reinterpret_cast<char*>(sig), 8);

        std::vector<unsigned char> ihdr;
        pushBE32(ihdr, (unsigned long)width);
        pushBE32(ihdr, (unsigned long)height);
        ihdr.push_back(8); // bit depth
        ihdr.push_back(2); // color type: 2 = truecolor (RGB)
        ihdr.push_back(0); // compression method
        ihdr.push_back(0); // filter method
        ihdr.push_back(0); // interlace method
        writeChunk(out, "IHDR", ihdr);
        writeChunk(out, "IDAT", idatData);
        writeChunk(out, "IEND", {});
    }
}

// Builds the black hole + N randomly placed extra bodies for one dataset sample.
// Extra bodies fully replace the two hardcoded demo planets used in interactive mode.
vector<ObjectData> buildDatasetObjects(float rs, double bhMass, int numExtraBodies, std::mt19937& rng) {
    vector<ObjectData> objs;
    objs.push_back({ vec4(0.0f, 0.0f, 0.0f, rs), vec4(0, 0, 0, 1), static_cast<float>(bhMass) });

    std::uniform_real_distribution<float> orbitDist(3.0f * rs, 15.0f * rs);
    std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * float(M_PI));
    std::uniform_real_distribution<float> heightDist(-2.0f * rs, 2.0f * rs);
    std::uniform_real_distribution<float> bodyRadiusDist(0.1f * rs, 0.6f * rs);
    std::uniform_real_distribution<float> colorDist(0.15f, 1.0f);
    std::uniform_real_distribution<double> bodyMassDist(1e28, 5e30);

    for (int b = 0; b < numExtraBodies; ++b) {
        float orbitR = orbitDist(rng);
        float ang    = angleDist(rng);
        float px = orbitR * cos(ang);
        float pz = orbitR * sin(ang);
        float py = heightDist(rng);

        ObjectData obj;
        obj.posRadius = vec4(px, py, pz, bodyRadiusDist(rng));
        obj.color     = vec4(colorDist(rng), colorDist(rng), colorDist(rng), 1.0f);
        obj.mass      = static_cast<float>(bodyMassDist(rng));
        objs.push_back(obj);
    }
    return objs;
}

// Interactive console flow: asks how many images, how many extra bodies, whether to
// randomize disk size, resolution and output directory, then renders each sample
// offscreen and writes an image + metadata row per sample.
void runDatasetGenerator() {
    cout << "\n=== Dataset Generator ===\n";

    long long numImages = 0;
    cout << "How many images do you want to generate? ";
    cin >> numImages;
    if (numImages <= 0) {
        cout << "Nothing to generate.\n";
        return;
    }

    int numExtraBodies = 0;
    cout << "How many extra celestial bodies in the scene (0 = none)? ";
    cin >> numExtraBodies;
    if (numExtraBodies < 0) numExtraBodies = 0;
    if (numExtraBodies > 15) {
        cout << "Note: the shader supports at most 16 objects total (black hole + extras),\n"
                "      so only the first 15 extra bodies will actually be rendered.\n";
    }

    cout << "Randomize accretion disk size per image, or keep it fixed? [r/f] (default f): ";
    char diskChoice = 'f';
    cin >> diskChoice;
    bool randomizeDisk = (diskChoice == 'r' || diskChoice == 'R');

    int resW = 400, resH = 300;
    cout << "Render width  [default " << resW << ", 0 = keep default]: ";
    int customW = 0; cin >> customW;
    if (customW > 0) {
        int customH = 0;
        cout << "Render height [default " << resH << "]: ";
        cin >> customH;
        if (customH > 0) { resW = customW; resH = customH; }
    }

    cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    string outDir = "dataset";
    cout << "Output directory [default '" << outDir << "']: ";
    string dirInput;
    std::getline(cin, dirInput);
    if (!dirInput.empty()) outDir = dirInput;
    std::filesystem::create_directories(outDir);

    // Moderate mass range: 0.5x - 2x Sagittarius A*'s mass. Edit here if you want a
    // wider/narrower spread of ring sizes.
    const double baseMass = 8.54e36;
    const double massMin  = 0.5 * baseMass;
    const double massMax  = 2.0 * baseMass;
    const double rsMax    = 2.0 * G * massMax / (c * c);

    const float fixedDiskR1Mult = 2.2f, fixedDiskR2Mult = 5.2f;
    const float diskR1MinMult = 2.0f, diskR1MaxMult = 3.0f;
    const float diskR2MinMult = 4.5f, diskR2MaxMult = 6.5f;

    // Fixed camera distance for the whole dataset: big enough that even the largest
    // disk (at massMax, and the largest randomized disk ratio if enabled) stays
    // comfortably inside the frame, with margin. FOV stays at the default 60 degrees.
    float maxDiskR2 = static_cast<float>(rsMax) * (randomizeDisk ? diskR2MaxMult : fixedDiskR2Mult);
    float tanHalfFov = tan(radians(60.0f * 0.5f));
    float camDistance = maxDiskR2 / (0.55f * tanHalfFov); // disk fills ~55% of half-frame at max size
    camDistance = std::max(camDistance, camera.minRadius);
    camera.radius = camDistance;

    cout << "Fixed camera distance: " << camDistance << " m\n";
    cout << "Generating " << numImages << " image(s) into '" << outDir << "'...\n";

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> massDist(massMin, massMax);
    std::uniform_real_distribution<float>  azimuthDist(0.0f, 2.0f * float(M_PI));
    std::uniform_real_distribution<float>  elevDist(0.15f * float(M_PI), 0.85f * float(M_PI)); // avoid the exact poles
    std::uniform_real_distribution<float>  diskR1MultDist(diskR1MinMult, diskR1MaxMult);
    std::uniform_real_distribution<float>  diskR2MultDist(diskR2MinMult, diskR2MaxMult);

    std::ofstream meta(outDir + "/labels.csv");
    meta << "filename,bh_mass_kg,schwarzschild_radius_m,camera_azimuth_rad,camera_elevation_rad,"
            "camera_distance_m,disk_r1_m,disk_r2_m,num_extra_bodies\n";

    glfwHideWindow(engine.window); // no need to show a live window while batch-rendering

    for (long long i = 0; i < numImages; ++i) {
        double mass = massDist(rng);
        setBlackHoleMass(mass);
        float rs = static_cast<float>(SagA.r_s);

        camera.azimuth   = azimuthDist(rng);
        camera.elevation = elevDist(rng);
        camera.update();

        float r1, r2;
        if (randomizeDisk) {
            r1 = rs * diskR1MultDist(rng);
            r2 = rs * diskR2MultDist(rng);
        } else {
            r1 = rs * fixedDiskR1Mult;
            r2 = rs * fixedDiskR2Mult;
        }

        vector<ObjectData> sampleObjects = buildDatasetObjects(rs, mass, numExtraBodies, rng);

        engine.renderDatasetSample(camera, resW, resH, r1, r2, rs, sampleObjects);

        std::vector<unsigned char> pixels(static_cast<size_t>(resW) * resH * 4);
        glBindTexture(GL_TEXTURE_2D, engine.texture);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

        char fname[64];
        snprintf(fname, sizeof(fname), "bh_%06lld.png", i);
        PNGWriter::write(outDir + "/" + fname, resW, resH, pixels);

        meta << fname << ',' << mass << ',' << SagA.r_s << ','
             << camera.azimuth << ',' << camera.elevation << ',' << camera.radius << ','
             << r1 << ',' << r2 << ',' << numExtraBodies << '\n';

        if (numExtraBodies > 0) {
            std::ofstream bodyJson(outDir + "/" + string(fname) + ".bodies.json");
            bodyJson << "[\n";
            for (size_t b = 1; b < sampleObjects.size(); ++b) {
                auto& o = sampleObjects[b];
                bodyJson << "  {\"x\":" << o.posRadius.x << ",\"y\":" << o.posRadius.y
                         << ",\"z\":" << o.posRadius.z << ",\"radius\":" << o.posRadius.w
                         << ",\"mass\":" << o.mass
                         << ",\"color\":[" << o.color.r << ',' << o.color.g << ',' << o.color.b << "]}";
                bodyJson << (b + 1 < sampleObjects.size() ? ",\n" : "\n");
            }
            bodyJson << "]\n";
        }

        if ((i + 1) % 10 == 0 || i == numImages - 1) {
            cout << "  [" << (i + 1) << "/" << numImages << "]\r" << std::flush;
        }
        glfwPollEvents(); // keep the (hidden) window from looking unresponsive
    }

    cout << "\nDone. " << numImages << " image(s) + labels.csv written to '" << outDir << "'\n";
}
// ================== END DATASET GENERATOR ==================== //

// -- MAIN -- //
int main() {
    setupCameraCallbacks(engine.window);

    cout << "Run (i)nteractive simulator or (d)ataset generator? [i/d] (default i): ";
    string modeInput;
    std::getline(cin, modeInput);
    if (!modeInput.empty() && (modeInput[0] == 'd' || modeInput[0] == 'D')) {
        runDatasetGenerator();
        glfwDestroyWindow(engine.window);
        glfwTerminate();
        return 0;
    }

    vector<unsigned char> pixels(engine.WIDTH * engine.HEIGHT * 3);

    auto t0 = Clock::now();
    lastPrintTime = chrono::duration<double>(t0.time_since_epoch()).count();

    double lastTime = glfwGetTime();
    int   renderW  = 800, renderH = 600, numSteps = 80000;
    while (!glfwWindowShouldClose(engine.window)) {
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);  // optional, but good practice
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        double now   = glfwGetTime();
        double dt    = now - lastTime;   // seconds since last frame
        lastTime     = now;

        // Gravity
        for (auto& obj : objects) {
            for (auto& obj2 : objects) {
                if (&obj == &obj2) continue; // skip self-interaction
                 float dx  = obj2.posRadius.x - obj.posRadius.x;
                 float dy = obj2.posRadius.y - obj.posRadius.y;
                 float dz = obj2.posRadius.z - obj.posRadius.z;
                 float distance = sqrt(dx * dx + dy * dy + dz * dz);
                 if (distance > 0) {
                        vector<double> direction = {dx / distance, dy / distance, dz / distance};
                        //distance *= 1000;
                        double Gforce = (G * obj.mass * obj2.mass) / (distance * distance);

                        double acc1 = Gforce / obj.mass;
                        std::vector<double> acc = {direction[0] * acc1, direction[1] * acc1, direction[2] * acc1};
                        if (Gravity) {
                            obj.velocity.x += acc[0];
                            obj.velocity.y += acc[1];
                            obj.velocity.z += acc[2];

                            obj.posRadius.x += obj.velocity.x;
                            obj.posRadius.y += obj.velocity.y;
                            obj.posRadius.z += obj.velocity.z;
                            cout << "velocity: " <<obj.velocity.x<<", " <<obj.velocity.y<<", " <<obj.velocity.z<<endl;
                        }
                    }
            }
        }



        // ---------- GRID ------------- //
        // 2) rebuild grid mesh on CPU
        engine.generateGrid(objects);
        // 5) overlay the bent grid
        mat4 view = lookAt(camera.position(), camera.target, vec3(0,1,0));
        mat4 proj = perspective(radians(60.0f), float(engine.COMPUTE_WIDTH)/engine.COMPUTE_HEIGHT, 1e9f, 1e14f);
        mat4 viewProj = proj * view;
        engine.drawGrid(viewProj);

        // ---------- RUN RAYTRACER ------------- //
        glViewport(0, 0, engine.WIDTH, engine.HEIGHT);
        engine.dispatchCompute(camera);
        engine.drawFullScreenQuad();

        // 6) present to screen
        glfwSwapBuffers(engine.window);
        glfwPollEvents();
    }

    glfwDestroyWindow(engine.window);
    glfwTerminate();
    return 0;
}
