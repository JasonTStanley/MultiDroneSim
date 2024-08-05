import os


def generate_sphere(radius):
    # create urdf text and write it to a file in tmp dir
    file_name = f"/tmp/sphere_{radius}.urdf"
    #check if file already exists
    if os.path.exists(file_name):
        return file_name
    urdf_text = f"""<?xml version="1.0" encoding="utf-8"?>
                    <robot name="sphere">
                      <link name="base_link">
                        <visual>
                          <geometry>
                            <sphere radius="{radius}"/>
                          </geometry>
                          <material name="blue"/>
                        </visual>
                        <collision>
                          <geometry>
                            <sphere radius="{radius}"/>
                          </geometry>
                        </collision>
                      </link>
                    </robot> """

    # write to file
    with open(file_name, "w") as f:
        f.write(urdf_text)
    return file_name



if __name__ == "__main__":
    generate_sphere(0.5)
