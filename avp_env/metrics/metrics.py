from avp_env.metrics.utils import get_result_tags

def calculate_success_rate(experiments):

    success_count = 0
    error_count = 0
    total_experiments = len(experiments)
    for experiment in experiments:
        result_id = experiment["result_id"]
        target_id = experiment["target_id"]
        if result_id and result_id in target_id:
            success_count += 1
        else:
            error_count += 1

    success_rate = success_count / total_experiments

    return success_rate


def calculate_weighted_success_rate(experiments):

    weighted_success_sum = 0
    total_experiments = len(experiments)
    for experiment in experiments:
        result_id = experiment["result_id"]
        target_id = experiment["target_id"]
        distance = experiment["result_features"]["distance"]

        if result_id and result_id in target_id:
            weighted_success_sum += 1 / distance

    weighted_success_rate = weighted_success_sum / total_experiments

    return weighted_success_rate


def calculate_navigation_errors(experiments):

    error_count = 0
    total_experiments = len(experiments)

    for experiment in experiments:
        if experiment["target_features"]:
            result_id = experiment["result_id"]
            result_path = experiment["target_features"][0]["scan"]
            result_park = experiment["target_features"][0]['park_id']

            if not result_id:
                continue
            result_tags = get_result_tags(result_id, result_park, result_path)
            if result_tags["Occupied"] != 0:
                error_count += 1



    error_rate = error_count / total_experiments

    return error_rate


def calculate_absolute_parking_slot_error(experiments):

    total_error = 0

    total_experiments = len(experiments)
    for experiment in experiments:
        target_positions = [target["distance"] for target in experiment["target_features"]]
        result_position = experiment["result_features"]["distance"]

        if not result_position and target_positions:
            min_distance = min(abs((87 - target_position)/ 87) for target_position in target_positions)
        elif not target_positions and result_position:
            min_distance = min(abs((result_position - 87)/ 87), abs(result_position/ 87))
        elif not target_positions and not result_position:
            min_distance = 0
        elif target_positions and result_position:
            min_distance = min(abs((result_position - target_position) / 87) for target_position in target_positions)

        total_error += min_distance

    apse = total_error / total_experiments

    return apse


def calculate_miss_rate(experiments):

    total_miss = 0
    total_experiments = len(experiments)

    for experiment in experiments:


        target_positions = [target["distance"] for target in experiment["target_features"]]
        result_position = experiment["result_features"]["distance"]


        if target_positions and result_position:
            targets_before_result = sum(1 for target_position in target_positions if target_position < result_position)
            # equivalent_miss = targets_before_result / len(target_positions)
            equivalent_miss = targets_before_result / len(target_positions)
            total_miss += equivalent_miss


    miss_rate = total_miss / total_experiments if total_experiments > 0 else 0

    return miss_rate


def calculate_advance_rate(experiments):

    total_advance = 0
    total_experiments = len(experiments)

    for experiment in experiments:
        target_positions = [target["distance"] for target in experiment["target_features"]]
        result_position = experiment["result_features"]["distance"]
        if not result_position and target_positions:
            total_advance += 0
        elif not target_positions and result_position:
            total_advance += 1
        elif not target_positions and not result_position:
            total_advance += 0
        elif all(result_position < target_position for target_position in target_positions):
            total_advance += 1

    advance_rate = total_advance / total_experiments if total_experiments > 0 else 0

    return advance_rate


def calculate_matching_rate(experiments):

    total_matching_score = 0
    total_experiments = len(experiments)

    for experiment in experiments:
        if experiment["target_features"]:

            result_id = experiment["result_id"]
            result_path = experiment["target_features"][0]["scan"]
            result_park = experiment["target_features"][0]['park_id']

            if not result_id:
                continue
            target_tags = experiment["target_features"][0]['tags']
            result_tags = get_result_tags(result_id, result_park, result_path)

            matching_tags = 0
            for tag, value in target_tags.items():
                if result_tags.get(tag) == value:
                    matching_tags += 1

            total_tags = len(target_tags)

            matching_score = matching_tags / total_tags if total_tags > 0 else 0

            total_matching_score += matching_score

    weighted_matching_rate = total_matching_score / total_experiments if total_experiments > 0 else 0

    return weighted_matching_rate

def calculate_weighted_matching_rate(experiments):

    total_matching_score = 0
    total_experiments = len(experiments)

    for experiment in experiments:
        if experiment["target_features"]:

            result_id = experiment["result_id"]
            result_path = experiment["target_features"][0]["scan"]
            result_park = experiment["target_features"][0]['park_id']
            if not result_id:
                continue
            target_tags = experiment["target_features"][0]['tags']
            result_tags = get_result_tags(result_id, result_park, result_path)
            distance = experiment["result_features"]["distance"]

            matching_tags = 0
            for tag, value in target_tags.items():
                if result_tags.get(tag) == value:
                    matching_tags += 1

            total_tags = len(target_tags)

            matching_score = matching_tags / total_tags / distance if total_tags > 0 else 0

            total_matching_score += matching_score

    weighted_matching_rate = total_matching_score / total_experiments if total_experiments > 0 else 0

    return weighted_matching_rate

def calculate_distance(experiments):

    total_result_position = 0
    total_experiments = len(experiments)

    for experiment in experiments:

        result_position = experiment["result_features"]["distance"]


        total_result_position += result_position

    result_rate = total_result_position / total_experiments if total_experiments > 0 else 0

    return result_rate


def get_parking_metrics(experiments):
    NE_Metrics = calculate_navigation_errors(experiments)
    SR_Metrics = calculate_success_rate(experiments)
    SRL_Metrics = calculate_weighted_success_rate(experiments)
    APSE_Metrics = calculate_absolute_parking_slot_error(experiments)
    MR_Metrics = calculate_miss_rate(experiments)
    CSR_Metrics = calculate_matching_rate(experiments)
    result_dis = calculate_distance(experiments)

    return {"CR": NE_Metrics, "SR": SR_Metrics, "TE": SRL_Metrics, "GDE": APSE_Metrics, "MR": MR_Metrics, "IGA": CSR_Metrics, "DIS": result_dis}
